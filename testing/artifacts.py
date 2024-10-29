from flytekit.configuration import Config, PlatformConfig, AuthType
from unionai.remote import UnionRemote

remote = UnionRemote(Config.auto().with_params(platform=PlatformConfig(
    endpoint="demo.hosted.unionai.cloud",
    insecure=False)))

remote.get_artifact("flyte://av0.2/demo/flytesnacks/development/NeRF@avtmflr6qmj64zr8zw8n/n4/0/o0")
