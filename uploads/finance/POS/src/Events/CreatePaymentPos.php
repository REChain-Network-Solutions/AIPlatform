<?php

namespace Workdo\Pos\Events;

use Illuminate\Queue\SerializesModels;

class CreatePaymentPos
{
    use SerializesModels;

    /**
     * Create a new event instance.
     *
     * @return void
     */
    public $request;
    public $pos;
    public $posPayment;
    public function __construct($pos,$posPayment,$request)
    {
        $this->request = $request;
        $this->pos = $pos;
        $this->posPayment = $posPayment;
    }

    /**
     * Get the channels the event should be broadcast on.
     *
     * @return array
     */
    public function broadcastOn()
    {
        return [];
    }
}
