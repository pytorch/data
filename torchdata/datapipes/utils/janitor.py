from torchdata.datapipes.utils import StreamWrapper


def janitor(obj):
    """
    Invokes various `obj` cleanup procedures such as:
    - Closing streams
    """
    # TODO(VitalyFedyunin): We can also release caching locks here to allow filtering
    StreamWrapper.close_streams(obj)
