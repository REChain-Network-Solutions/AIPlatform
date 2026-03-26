<?php

namespace Modules\Inspection\Policies;

use App\Models\User;

class TemplatePolicy
{
    public function viewAny(User $user): bool { return true; }
    public function view(User $user): bool { return true; }
    public function create(User $user): bool { return true; }
    public function update(User $user): bool { return true; }
    public function delete(User $user): bool { return true; }
}
