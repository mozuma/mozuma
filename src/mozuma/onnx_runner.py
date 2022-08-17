import numpy as np
import torch
import logging
import tempfile
import onnxruntime as ort

from tqdm import tqdm
from typing import Tuple
from mozuma.torch.runners import TorchInferenceRunner
from mozuma.torch.utils import send_batch_to_device
from mozuma.torch.callbacks import TorchRunnerCallbackType
from mozuma.callbacks.base import callbacks_caller

logger = logging.getLogger(__name__)

class ONNXInferenceRunner(TorchInferenceRunner):
  """Runner for inference tasks on ONNX models

    Supports CPU or single GPU inference.

    Attributes:
        model: The PyTorch model to run inference
        dataset: Input dataset for the runner
        callbacks: Callbacks to save features, labels or bounding boxes
        options: PyTorch options
    """

  def __init__(self, model, dataset, callbacks, options):
    super(TorchInferenceRunner, self).__init__(model, dataset, callbacks, options)
    self.device_name = str(self.options.device)


  def transfer(self, onnx_file: str) -> None:
    """Transfers PyTorch model to ONNX form"""

    # setting model in eval mode
    self.model.eval()

    # sending model on device
    self.model.to(self.device_name)

    # specifying the shapes of an input 
    dummy_input = next(iter(self.get_data_loader()))[1].to(self.device_name)
    output = self.model(dummy_input)
    self.outputs_count = 1 if type(output) == torch.Tensor else len(output)

    # setting input and output names for ONNX model
    self.input_names = [ "input" ]
    self.output_names = [ "output" + str(output_id) for output_id in range(self.outputs_count)]

    # exporting PyTorch model to ONNX
    dynamic_axes = {dynamic_value: {0: "batch_size"} for dynamic_value in self.input_names + self.output_names}
    torch.onnx.export(self.model, 
                      dummy_input, 
                      onnx_file, 
                      input_names=self.input_names, 
                      output_names=self.output_names, 
                      dynamic_axes = dynamic_axes,
                      opset_version=16)

  
  def session_setup(
      self,
      tmp_onnx: tempfile.NamedTemporaryFile
  ) -> Tuple[ort.InferenceSession, ort.IOBinding]:
    """Set up onnx and iobinding sessions"""

    # specify onnx session options and create inference session
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.inter_op_num_threads = 2
    so.execution_mode = ort.ExecutionMode.ORT_PARALLEL
    session = ort.InferenceSession(tmp_onnx.name, sess_options=so, 
                                    providers=[("CUDAExecutionProvider", {"cudnn_conv_use_max_workspace": '1'}), 'CPUExecutionProvider'])

    # warm up run
    dummy_input = next(iter(self.get_data_loader()))[1].to(self.device_name)
    input_name = session.get_inputs()[0].name
    session.run([], {input_name: dummy_input.cpu().numpy()})

    # get output shapes and dtypes
    outputs = self.model(dummy_input)
    outputs = [outputs] if type(outputs) == torch.Tensor else outputs
    self.output_shapes, self.output_dtypes  = [], []
    for output in outputs:
      self.output_shapes.append(output.size()[1])
      dtype = eval(str(output.dtype).replace('torch', 'np'))
      self.output_dtypes.append(dtype)
      
      self.input_dtype = eval(str(dummy_input.dtype).replace('torch', 'np'))

    return session, session.io_binding()


  def run_iobinding(
      self, 
      batch: torch.Tensor,
      session: ort.InferenceSession,
      iobinding: ort.IOBinding
  ) -> ort.OrtValue:
    """Running inference with iobinding"""

    X_ortvalue = ort.OrtValue.ortvalue_from_numpy(batch.numpy(), self.device_name, 0)
    iobinding.bind_input(name='input', device_type=X_ortvalue.device_name(), device_id=0,
                          element_type=self.input_dtype, shape=X_ortvalue.shape(), buffer_ptr=X_ortvalue.data_ptr())
    
    Y_ortvalues = []
    for output_id in range(self.outputs_count):
      Y_ortvalue = ort.OrtValue.ortvalue_from_shape_and_type([batch.shape[0], self.output_shapes[output_id]], self.output_dtypes[output_id], self.device_name, 0)
      Y_ortvalues.append(Y_ortvalue)
      iobinding.bind_output(name=self.output_names[output_id], device_type=Y_ortvalue.device_name(), device_id=0,
                              element_type=self.output_dtypes[output_id], shape=Y_ortvalue.shape(), buffer_ptr=Y_ortvalue.data_ptr())
      
    # Running inference
    session.run_with_iobinding(iobinding)

    return Y_ortvalues


  def run(self) -> None:
    """Runs inference"""

    # create temporary file for storing ONNX model
    with tempfile.NamedTemporaryFile() as tmp_onnx:

      # convert PyTorch model to ONNX
      self.transfer(tmp_onnx.name)

      # disabling gradient computation
      with torch.no_grad():
        # building data loader
        data_loader = self.get_data_loader()
        
        # set up onnx and iobinding sessions
        onnx_session, iobinding_session = self.session_setup(tmp_onnx)

        # Looping through batches
        # Assume dataset is composed of tuples (item index, batch)
        n_batches = len(data_loader)
        loader = tqdm(data_loader) if self.options.tqdm_enabled else data_loader
        for batch_n, (indices, batch) in enumerate(loader):
            logger.debug(f"Sending batch number: {batch_n}/{n_batches}")  

            # running inference with iobinding
            Y = self.run_iobinding(batch, onnx_session, iobinding_session)

            # convert ortvalues to numpy and get predictions
            Y_numpy = [Y_ortvalue.numpy() for Y_ortvalue in Y]
            Y_numpy = Y_numpy[0] if self.outputs_count == 1 else Y_numpy
            predictions = self.model.to_predictions(Y_numpy)

            # Applying callbacks on results
            self.apply_predictions_callbacks(indices, predictions)
            logger.debug(f"Collecting results: {batch_n}/{n_batches}")


    # Notify the end of the runner
    callbacks_caller(self.callbacks, "on_runner_end", self.model)
