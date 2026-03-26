<?php

namespace Modules\Inspection\Policies;

use App\Models\User;

class SchedulePolicy
{
    public function viewAny(User $user): bool
    {
        return $user->hasPermissionTo('inspection.view');
    }
}
