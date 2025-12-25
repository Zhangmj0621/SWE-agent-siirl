from .base import ContainerEnv, ContainerEnvBuilder, ContainerStartArgs


# TODO: docker container, using https://github.com/docker/docker-py
class DockerEnv(ContainerEnv):
    pass


class DockerEnvBuilder(ContainerEnvBuilder):
    async def start(self, args: ContainerStartArgs) -> DockerEnv:
        raise NotImplementedError
