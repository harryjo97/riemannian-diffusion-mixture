from abc import ABC, abstractmethod

class BetaSchedule(ABC):
    @abstractmethod
    def beta_t(self, t):
        pass

    @abstractmethod
    def reverse(self):
        pass
        

class LinearBetaSchedule(BetaSchedule):
    def __init__(
        self,
        tf: float = 1,
        t0: float = 0,
        beta_0: float = 0.2,
        beta_f: float = 0.001,
    ):
        self.tf = tf
        self.t0 = t0
        self.beta_0 = beta_0
        self.beta_f = beta_f
        self._beta = beta_f - beta_0
        self._t = tf - t0

        self.normed = (t0==0.0 and tf==1.0)

    def normed_t(self, t):
        return (t - self.t0) / self._t

    def rescale_t(self, t):
        normed_t = self.normed_t(t)
        return (t - t0) * self.beta_0 + 0.5 * (t - t0)**2  * self._beta / self._t

    def rescale_t_delta(self, s, t):
        dt = t - s
        if self.normed:
            return dt * (self.beta_0 + 0.5 * (t+s) * self._beta)
        else:
            return dt * self.beta_0 + (0.5 * (t+s) - self.t0) * self._beta * dt / self._t

    def beta_t(self, t):
        normed_t = self.normed_t(t)
        return self.beta_0 + normed_t * self._beta

    def reverse(self):
        return LinearBetaSchedule(
            tf=self.tf, t0=self.t0, beta_f=self.beta_0, beta_0=self.beta_f
        )