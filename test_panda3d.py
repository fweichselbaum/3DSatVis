from direct.showbase.ShowBase import ShowBase


class Debug(ShowBase):
	def __init__(self):
		ShowBase.__init__(self)

		axis = self.loader.loadModel("models/zup-axis")
		axis.reparentTo(self.render)

		self.messenger.toggleVerbose()


Debug().run()