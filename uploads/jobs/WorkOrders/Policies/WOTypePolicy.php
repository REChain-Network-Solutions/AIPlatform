<?php

namespace Modules\WorkOrders\Policies;

use App\Models\User;
use Modules\WorkOrders\Entities\WOType;

class WOTypePolicy
{
    protected \Modules\WorkOrders\Policies\WorkOrderPolicy $base;

    public function __construct()
    {
        $this->base = new \Modules\WorkOrders\Policies\WorkOrderPolicy();
    }

    public function viewAny(User $user): bool { return $this->base->viewAny($user); }
    public function view(User $user, WOType $m): bool { return $this->base->view($user, $m->workOrder ?? $m); }
    public function create(User $user): bool { return $this->base->create($user); }
    public function update(User $user, WOType $m): bool { return $this->base->update($user, $m->workOrder ?? $m); }
    public function delete(User $user, WOType $m): bool { return $this->base->delete($user, $m->workOrder ?? $m); }
    public function manageSettings(User $user): bool { return $this->base->manageSettings($user); }
}
