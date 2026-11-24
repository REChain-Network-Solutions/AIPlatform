<?php

use Illuminate\Database\Migrations\Migration;

return new class extends Migration
{
    /**
     * `total_spend` was previously written with number_format($value, 2), which inserts a
     * thousands separator once the value passes 1000. The stored string ("1,006.63") then
     * broke the next arithmetic read with "A non-numeric value encountered".
     */
    public function up(): void
    {
        $value = setting('total_spend');

        if ($value === null) {
            return;
        }

        $normalized = number_format((float) str_replace(',', '', (string) $value), 2, '.', '');

        setting(['total_spend' => $normalized])->save();
    }

    public function down(): void
    {
        //
    }
};
