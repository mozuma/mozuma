# Models

## Pre-trained models

| Reference | Name | Purposes | License |
| --------- | ---- | ------- | -------- |{% for module in models -%}
{% for model in models[module] %}
| [`{{ model.factory }}`][mozuma.models.{{ module }}.{{ model.factory }}] | {{ model.name }} | {{ model.purposes | join("<br>") }} | [{{ licenses[model.license].name }} :octicons-link-external-16:]({{ licenses[model.license].link }}){:target="\_blank"} |{% endfor -%}
{% endfor -%}
