<?php

namespace Modules\Inspection\Policies;

use App\Models\User;

class InspectionPolicy
{
    public function viewAny(User $user): bool
    {
        return $user->hasPermissionTo('inspection.view');
    }

    public function create(User $user): bool
    {
        return $user->hasPermissionTo('inspection.create');
    }
}
