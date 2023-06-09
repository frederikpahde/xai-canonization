import quantus

def wrap_zennit_quantus(canonizer, model, inputs, targets, device, *args, **kwargs):
    if kwargs.get('is_vqa', False):
        q = kwargs['question']
        len_q = kwargs['q_length']
        inputs = (inputs, q, len_q)

    attribution = quantus.explain(
        model.eval(), 
        inputs, 
        targets, 
        canonizer=canonizer,
        device=device,
        method="custom",
        *args, 
        **kwargs
    )
    return attribution
