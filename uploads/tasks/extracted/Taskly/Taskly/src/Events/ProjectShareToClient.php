<?php

namespace Workdo\Taskly\Events;

use Illuminate\Queue\SerializesModels;

class ProjectShareToClient
{
    use SerializesModels;

    /**
     * Create a new event instance.
     *
     * @return void
     */
    public $request;
    public $client;
    public $project;

    public function __construct($request,$client,$project)
    {
        $this->request  = $request;
        $this->client   = $client;
        $this->project  = $project;
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
