from .objects import Object


class Scene:
    def __init__(self):
        self.objects = []

    def add(self, item):
        if isinstance(item, Object):
            self.add_object(item)

    def add_object(self, obj):
        self.objects.append(obj)