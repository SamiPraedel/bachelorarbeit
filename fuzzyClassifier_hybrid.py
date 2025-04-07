from manim import *

class Blue1BrownStyleScene(Scene):
    def construct(self):
        # Title Animation: Display a title at the top of the screen.
        title = Text("3Blue1Brown Style Animation", font_size=60)
        title.to_edge(UP)
        self.play(Write(title))
        self.wait(1)
        
        # Geometric Transformation: Create a circle and transform it into a square.
        circle = Circle(radius=1.5, color=BLUE)
        square = Square(side_length=3, color=GREEN)
        square.move_to(circle.get_center())
        
        self.play(Create(circle))
        self.wait(1)
        self.play(Transform(circle, square))
        self.wait(1)
        
        # Graph Animation: Create axes and draw a sine wave.
        axes = Axes(
            x_range=[0, 10, 1],
            y_range=[-2, 2, 1],
            x_length=10,
            y_length=4,
            tips=False
        )
        sine_wave = axes.get_graph(lambda x: np.sin(x), color=RED)
        self.play(Create(axes), run_time=2)
        self.play(Create(sine_wave), run_time=2)
        
        # Dot Animation: Animate a dot moving along the sine wave.
        dot = Dot(color=YELLOW).move_to(axes.c2p(0, np.sin(0)))
        self.play(FadeIn(dot))
        self.wait(1)
        self.play(MoveAlongPath(dot, sine_wave), run_time=4, rate_func=linear)
        self.wait(1)
        
        # Fade out all elements to conclude the scene.
        self.play(FadeOut(title), FadeOut(square), FadeOut(axes), FadeOut(dot))