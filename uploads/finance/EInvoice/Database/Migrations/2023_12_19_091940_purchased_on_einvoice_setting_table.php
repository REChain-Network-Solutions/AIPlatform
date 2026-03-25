<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

return new class extends Migration
{
    /**
     * Run the migrations.
     */

    public function up(): void
    {

        Schema::whenTableDoesntHaveColumn('e_invoice_settings', 'purchased_on', function (Blueprint $table) {
            $table->timestamp('purchased_on')->nullable()->after('supported_until');
        });
    }

    /**
     * Reverse the migrations.
     */
    public function down(): void
    {
        Schema::table('order_items', function (Blueprint $table) {

        });
    }

};
