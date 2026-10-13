import uuid
import pytest
from pydantic import ValidationError
from people_gator_common.people_gator_models import TextMentionPatchRequest, FacePatchRequest, EntityListFilter
from people_gator_common.client.pydantic_gen import TextMentionPatchRequest as ClientMentionPatch


@pytest.mark.parametrize('model', [TextMentionPatchRequest, FacePatchRequest, ClientMentionPatch])
def test_state_only_patch_keeps_omission_and_unlink_distinct(model):
    assert model(entity_link_state='verified').model_dump(mode='json') == {'entity_link_state': 'verified'}
    assert model(entity_id=None).model_dump(mode='json', exclude_unset=True) == {'entity_id': None}
    with pytest.raises(ValidationError): model(entity_link_state='unknown')


def test_link_state_filter_is_serialized_with_entity_identity():
    entity_id = uuid.uuid4()
    payload = EntityListFilter(entity_id=[entity_id], entity_link_state=['automatic','verified']).model_dump(mode='json', exclude_unset=True)
    assert payload['entity_link_state'] == ['automatic','verified']
    assert payload['entity_id'] == [str(entity_id)]
