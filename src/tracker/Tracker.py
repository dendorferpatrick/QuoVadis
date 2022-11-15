from abc import ABC

class Tracker(ABC):
    def get_frame(self, frame):
        df_frame = self.df[self.df.frame == frame]
        return df_frame

    def get_pixel_coordinates(self): 
        self.df["y_pixel"] = self.df["bb_top"] + self.df["bb_height"]
        self.df["x_pixel"] = self.df["bb_left"] + self.df["bb_width"]/ 2
    def len(self, frame = None ):
        if frame is not None:
            return len(self.df[self.df.frame <= frame])
        return len(self.df)

    