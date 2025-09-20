<?php

namespace Database\Seeders;

use App\Enums\DashboardWidget;
use App\Models\DashboardWidget as ModelsDashboardWidget;
use Illuminate\Database\Seeder;

class DashboardWidgetSeeder extends Seeder
{
    /**
     * Run the database seeds.
     */
    public function run(): void
    {
        $dashboardWidgets = DashboardWidget::cases();
        foreach ($dashboardWidgets as $index => $case) {
            ModelsDashboardWidget::firstOrCreate([
                'name' => $case->value,
            ], [
                'enabled' => true,
                'order'   => $index,
            ]);
        }
    }
}
