<?php

namespace Workdo\Taskly\Database\Seeders;

use App\Models\Notification;
use Illuminate\Database\Seeder;

class NotificationsTableSeeder extends Seeder
{
    /**
     * Run the database seeds.
     *
     * @return void
     */
    public function run()
    {
        // email notification
        $notifications = [
            'User Invited',
            'Project Assigned',
        ];

        $permissions = [
            'taskly manage',
            'taskly manage',
        ];

        foreach ($notifications as $key => $n) {
            $ntfy = Notification::where('action', $n)->where('type', 'mail')->where('module', 'Taskly')->count();
            if ($ntfy == 0) {
                $new = new Notification();
                $new->action = $n;
                $new->status = 'on';
                $new->permissions = $permissions[$key];
                $new->module = 'Taskly';
                $new->type = 'mail';
                $new->save();
            }
        }

    }
}
