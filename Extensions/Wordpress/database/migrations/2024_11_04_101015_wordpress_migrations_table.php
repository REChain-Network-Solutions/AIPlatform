<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Support\Facades\Schema;

return new class extends Migration
{
    public function up(): void
    {
        if (Schema::hasTable('integrations')) {
            return;
        }

        if (App\Models\Integration\Integration::query()->where('slug', 'wordpress')->doesntExist()) {
            App\Models\Integration\Integration::query()->create([
                'app'         => 'Wordpress',
                'description' => 'Wordpress integration',
                'image'       => 'images/integrations/wordpress.png',
                'slug'        => 'wordpress',
                'status'      => 1,
            ]);
        }

        App\Models\Integration\Integration::query()->where('slug', 'wordpress')->update([
            'image'  => 'images/integrations/wordpress.png',
            'status' => 1,
        ]);
    }

    public function down(): void {}
};
