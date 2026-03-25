<?php

namespace Workdo\Taskly\Events;

use Illuminate\Queue\SerializesModels;

class ProjectUploadFiles
{
    use SerializesModels;

    /**
     * Create a new event instance.
     *
     * @return void
     */
    public $request;
    public $upload;
    public $project;

    public function __construct($request , $upload , $project)
    {
        $this->request  = $request;
        $this->upload   = $upload;
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
