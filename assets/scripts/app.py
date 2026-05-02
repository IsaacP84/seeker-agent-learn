import Magic
print("Loading app")
print(Magic.app)
        
sm = Magic.app.SceneManager()

rm = Magic.app.ResourceManager()

files = [
    "wall.jpg",
    "container.jpg",
    "container2.png",
    "container2_specular2.png",
    "matrix.jpg",
    "matrix-red.jpg",
]
for i, f in enumerate(files):
    files[i] = Magic.ASSETS_FOLDER + "/images/" + f

rm.load(files, Magic.Handle.Type.TEXTURE) 