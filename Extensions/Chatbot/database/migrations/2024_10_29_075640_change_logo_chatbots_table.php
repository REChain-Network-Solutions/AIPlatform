<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

return new class extends Migration
{
    public static $prefix = 'ext';

    public function up(): void
    {
        Schema::table(self::$prefix . '_chatbots', function (Blueprint $table) {
            $table->text('logo')->nullable()->comment('base 64')->change();
        });
    }

    public function down(): void {}
};
