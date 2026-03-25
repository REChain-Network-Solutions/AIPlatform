<?php

namespace Workdo\Taskly\Listeners;

use App\Events\GivePermissionToRole;
use Workdo\Taskly\Entities\TaskUtility;

class GiveRoleToPermission
{
    /**
     * Create the event listener.
     *
     * @return void
     */
    public function __construct()
    {
        //
    }

    /**
     * Handle the event.
     *
     * @param  object  $event
     * @return void
     */
    public function handle(GivePermissionToRole $event)
    {
        $role_id = $event->role_id;
        $rolename = $event->rolename;
        $user_module = $event->user_module;
        if(!empty($user_module))
        {
            if (in_array("Taskly", $user_module))
            {
                TaskUtility::GivePermissionToRoles($role_id,$rolename);
            }
        }
    }
}
