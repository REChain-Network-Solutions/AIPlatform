<?php

namespace Modules\PMCore\Database\Seeders;

use Illuminate\Database\Seeder;

class PMCoreDatabaseSeeder extends Seeder
{
    /**
     * Run the database seeds.
     */
    public function run(): void
    {
        $this->call([
            PMCorePermissionSeeder::class,
            ProjectStatusSeeder::class,
            ProjectSeeder::class,
            ProjectTaskSeeder::class,
            TimesheetSeeder::class,
            ResourceAllocationSeeder::class,
        ]);
    }
}
