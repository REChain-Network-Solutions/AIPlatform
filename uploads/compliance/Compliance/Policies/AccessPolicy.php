<?php
namespace Modules\Compliance\Policies;
use App\Models\User;
class AccessPolicy {
  public function access(?User $user): bool {
    if (!$user) return false;
    if (method_exists($user, 'can')) return $user->can('compliance.access');
    return property_exists($user, 'is_admin') ? (bool)$user->is_admin : true;
  }
}
