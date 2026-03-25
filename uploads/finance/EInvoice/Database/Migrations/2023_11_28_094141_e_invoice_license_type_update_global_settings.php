<?php

use Illuminate\Database\Schema\Blueprint;
use Illuminate\Database\Migrations\Migration;

return new class extends Migration
{

    /**
     * Run the migrations.
     */
    public function up(): void
    {
        Schema::whenTableDoesntHaveColumn('e_invoice_settings', 'license_type', function (Blueprint $table) {
            $table->string('license_type', 20)->nullable()->after('supported_until');
        });
    }

    /**
     * Reverse the migrations.
     */
    public function down(): void
    {
        //
    }

};
