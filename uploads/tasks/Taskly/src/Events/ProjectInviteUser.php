<?php

namespace Workdo\Taskly\Events;

use Illuminate\Queue\SerializesModels;

class ProjectInviteUser
{
    use SerializesModels;

    /**
     * Create a new event instance.
     *
     * @return void
     */

    public $request;
    public $registerUsers;
    public $objProject;

    public function __construct($request,$registerUsers,$objProject )
    {
        $this->request  = $request;
        $this->user     = $registerUsers;
        $this->project  = $objProject;

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
