import numpy as np

class CurriculumScheduler:
    """Управление расписанием SNR для эффективного обучения в области водопада."""
    def __init__(self, total_epochs):
        self.total = total_epochs

    def get_snr(self, epoch):
        if epoch < self.total * 0.3:
            return float(np.random.uniform(1.0, 3.0))   # Исследование
        elif epoch < self.total * 0.7:
            return float(np.random.uniform(3.0, 5.0))   # Waterfall
        else:
            return float(np.random.uniform(2.5, 6.0))   # Тонкая настройка