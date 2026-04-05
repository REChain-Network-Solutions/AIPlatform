<?php

namespace App\Support\ModuleDoctor;

class VisibilityProfileMatrixService
{
    /**
     * Returns a map of tenant profile names to their visibility constraints.
     */
    public function profiles(): array
    {
        return [
            'superadmin'   => ['requires_package' => false],
            'admin'        => ['requires_package' => true],
            'client'       => ['requires_package' => true],
            'team_member'  => ['requires_package' => true],
            'employee'     => ['requires_package' => true],
        ];
    }

    public function build(array $tenantResults): array
    {
        $matrix = [];
        foreach ($tenantResults as $profile => $result) {
            $matrix[] = [
                'profile' => $profile,
                'status' => $result['status'] ?? 'warn',
                'detail' => $result['detail'] ?? '',
            ];
        }

        return $matrix;
    }
}
