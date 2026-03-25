<?php

namespace Database\Seeders;

use App\Models\Common\Menu;
use App\Services\Common\MenuService;
use Illuminate\Database\Seeder;

/**
 * Seeds the Titan BOS CRM module navigation items into the menu table.
 * Run: php artisan db:seed --class=BosMenuSeeder
 */
class BosMenuSeeder extends Seeder
{
    public function run(): void
    {
        $items = $this->items();

        foreach ($items as $item) {
            Menu::query()->firstOrCreate(
                ['key' => $item['key']],
                [
                    'parent_id'      => $item['parent_key']
                        ? Menu::query()->where('key', $item['parent_key'])->value('id')
                        : null,
                    'route'          => $item['route'] ?? null,
                    'route_slug'     => $item['route_slug'] ?? null,
                    'label'          => $item['label'],
                    'icon'           => $item['icon'] ?? null,
                    'svg'            => $item['svg'] ?? null,
                    'order'          => $item['order'] ?? 99,
                    'is_active'      => $item['is_active'] ?? true,
                    'params'         => $item['params'] ?? [],
                    'type'           => $item['type'] ?? 'item',
                    'extension'      => $item['extension'] ?? null,
                    'letter_icon'    => $item['letter_icon'] ?? null,
                    'letter_icon_bg' => $item['letter_icon_bg'] ?? null,
                ]
            );
        }

        // Re-seed children now that parent IDs exist
        foreach ($items as $item) {
            if ($item['parent_key']) {
                Menu::query()->where('key', $item['key'])->update([
                    'parent_id' => Menu::query()->where('key', $item['parent_key'])->value('id'),
                ]);
            }
        }

        app(MenuService::class)->regenerate();
    }

    private function items(): array
    {
        return [
            // ── BOS section label ──────────────────────────────
            [
                'parent_key' => null,
                'key'        => 'bos_label',
                'route'      => null,
                'label'      => 'Business OS',
                'icon'       => null,
                'order'      => 50,
                'is_active'  => true,
                'type'       => 'label',
            ],

            // ── CRM parent item ───────────────────────────────
            [
                'parent_key'     => null,
                'key'            => 'crm',
                'route'          => 'dashboard.crm.dashboard',
                'label'          => 'CRM',
                'icon'           => 'tabler-users-group',
                'letter_icon'    => 'C',
                'letter_icon_bg' => '#6366f1',
                'order'          => 51,
                'is_active'      => true,
                'type'           => 'item',
            ],

            // CRM children
            [
                'parent_key' => 'crm',
                'key'        => 'crm_dashboard',
                'route'      => 'dashboard.crm.dashboard',
                'label'      => 'CRM Overview',
                'icon'       => 'tabler-layout-dashboard',
                'order'      => 0,
                'is_active'  => true,
                'type'       => 'item',
            ],
            [
                'parent_key' => 'crm',
                'key'        => 'crm_contacts',
                'route'      => 'dashboard.crm.contacts.index',
                'label'      => 'Contacts',
                'icon'       => 'tabler-users',
                'order'      => 1,
                'is_active'  => true,
                'type'       => 'item',
            ],
            [
                'parent_key' => 'crm',
                'key'        => 'crm_companies',
                'route'      => 'dashboard.crm.companies.index',
                'label'      => 'Companies',
                'icon'       => 'tabler-building',
                'order'      => 2,
                'is_active'  => true,
                'type'       => 'item',
            ],
            [
                'parent_key' => 'crm',
                'key'        => 'crm_deals',
                'route'      => 'dashboard.crm.deals.index',
                'label'      => 'Pipeline',
                'icon'       => 'tabler-layout-kanban',
                'order'      => 3,
                'is_active'  => true,
                'type'       => 'item',
            ],
            [
                'parent_key' => 'crm',
                'key'        => 'crm_activities',
                'route'      => 'dashboard.crm.activities.index',
                'label'      => 'Activities',
                'icon'       => 'tabler-activity',
                'order'      => 4,
                'is_active'  => true,
                'type'       => 'item',
            ],
            [
                'parent_key' => 'crm',
                'key'        => 'crm_leads',
                'route'      => 'dashboard.crm.leads.index',
                'label'      => 'Leads',
                'icon'       => 'tabler-user-plus',
                'order'      => 5,
                'is_active'  => true,
                'type'       => 'item',
            ],
        ];
    }
}
