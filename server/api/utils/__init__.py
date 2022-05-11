

class Meta:
    def __init__(self,
                 programmer:str,
                 main: str,
                 description: str,
                 language: str,
                 libraries: list
                 ):
        self.programmer = programmer
        self.main = main
        self.description = description
        self.language = language
        self.libraries= libraries
        
    def to_json(self):
        return{
            "programmer": self.programmer,
            "main": self.main,
            "description": self.description,
            "language": self.language,
            "libraries": self.libraries,
        }