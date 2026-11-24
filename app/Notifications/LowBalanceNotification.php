<?php

declare(strict_types=1);

namespace App\Notifications;

use Illuminate\Bus\Queueable;
use Illuminate\Contracts\Queue\ShouldQueue;
use Illuminate\Notifications\Notification;

class LowBalanceNotification extends Notification implements ShouldQueue
{
    use Queueable;

    public function __construct(
        private readonly float $currentBalance,
        private readonly float $planCredits,
    ) {}

    public function via(object $notifiable): array
    {
        return ['database'];
    }

    public function toArray(object $notifiable): array
    {
        $percentage = $this->planCredits > 0
            ? round(($this->currentBalance / $this->planCredits) * 100, 1)
            : 0;

        return [
            'title'      => __('Low Credit Balance'),
            'message'    => __('Your shared credit balance is low (:percent% remaining).', ['percent' => $percentage]),
            'balance'    => $this->currentBalance,
            'percentage' => $percentage,
        ];
    }
}
