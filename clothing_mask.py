from PIL import ImageOps
from .clothing_segmentation import ClothingSegmentation


from invokeai.app.services.image_records.image_records_common import ImageCategory, ResourceOrigin
from invokeai.app.invocations.baseinvocation import (
    BaseInvocation,
    InputField,
    invocation,
    InvocationContext,
    WithMetadata,
    WithWorkflow,
)

from invokeai.app.invocations.primitives import (
    ImageField,
    ImageOutput
)


@invocation(
    "clothing_mask",
    title="Clothing Mask",
    tags=["image", "mask"],
    category="image",
    version="1.0.0",
)
class ClothingMaskIvocation(BaseInvocation, WithMetadata, WithWorkflow):
    """Get a mask of clothes"""
    image: ImageField = InputField(default=None, description="Input image")

    def invoke(self, context: InvocationContext) -> ImageOutput:
        image = context.services.images.get_pil_image(self.image.image_name)

        segmentation = ClothingSegmentation()
        result = segmentation.segmentation(image.convert("RGB")).convert('RGBA')

        image_dto = context.services.images.create(
            image=result,
            image_origin=ResourceOrigin.INTERNAL,
            image_category=ImageCategory.GENERAL,
            node_id=self.id,
            session_id=context.graph_execution_state_id,
            is_intermediate=self.is_intermediate,
            workflow=self.workflow,
        )

        return ImageOutput(
            image=ImageField(image_name=image_dto.image_name),
            width=image_dto.width,
            height=image_dto.height,
        )
    

    