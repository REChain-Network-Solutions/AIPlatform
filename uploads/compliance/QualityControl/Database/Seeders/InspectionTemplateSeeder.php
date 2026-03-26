<?php

namespace Modules\QualityControl\Database\Seeders;

use Illuminate\Database\Seeder;
use Modules\QualityControl\Entities\InspectionTemplate;
use Modules\QualityControl\Entities\InspectionTemplateItem;

class InspectionTemplateSeeder extends Seeder
{
    public function run(): void
    {
        // Idempotent defaults for cleaning quality inspections.
        $templates = [
            [
                'name' => 'Standard Clean - Bathroom',
                'trade' => 'cleaning',
                'description' => 'Bathroom quality inspection checklist.',
                'items' => [
                    'Toilet cleaned & disinfected',
                    'Shower / bath cleaned (no soap scum)',
                    'Sink & taps polished',
                    'Mirror streak-free',
                    'Bins emptied & liner replaced',
                    'Floor vacuumed & mopped',
                ],
            ],
            [
                'name' => 'Standard Clean - Kitchen',
                'trade' => 'cleaning',
                'description' => 'Kitchen quality inspection checklist.',
                'items' => [
                    'Benches wiped & sanitised',
                    'Sink cleaned & drained',
                    'Appliance exteriors wiped',
                    'Splashback cleaned',
                    'Bins emptied & liner replaced',
                    'Floor vacuumed & mopped',
                ],
            ],
            [
                'name' => 'End of Lease - Final QC',
                'trade' => 'cleaning',
                'description' => 'End of lease final quality control checklist.',
                'items' => [
                    'All rooms dusted & surfaces wiped',
                    'Walls spot-cleaned (where applicable)',
                    'Windows/glass cleaned (as per scope)',
                    'Bathroom deep-clean complete',
                    'Kitchen deep-clean complete',
                    'Floors vacuumed & mopped',
                    'Rubbish removed',
                ],
            ],
        ];

        foreach ($templates as $t) {
            $template = InspectionTemplate::firstOrCreate(
                ['name' => $t['name']],
                [
                    'trade' => $t['trade'],
                    'description' => $t['description'],
                    'is_active' => true,
                ]
            );

            // Ensure items exist (idempotent by name per template)
            foreach ($t['items'] as $index => $itemName) {
                InspectionTemplateItem::firstOrCreate(
                    ['template_id' => $template->id, 'item_name' => $itemName],
                    ['sort_order' => $index + 1]
                );
            }
        }
    }
}
