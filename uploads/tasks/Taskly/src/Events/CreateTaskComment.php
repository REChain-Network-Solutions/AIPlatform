<?php

namespace Workdo\Taskly\Events;

use Illuminate\Queue\SerializesModels;

class CreateTaskComment
{
    use SerializesModels;

    /**
     * Create a new event instance.
     *
     * @return void
     */
    public $request;
    public $comment;

    public function __construct($request,$comment)
    {
        $this->request = $request;
        $this->comment = $comment;
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
