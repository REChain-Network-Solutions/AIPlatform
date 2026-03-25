<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

return new class extends Migration
{
    public function up(): void
    {
        if (!Schema::hasTable('einvoice_invoice_items')) {
            Schema::create('einvoice_invoice_items', function (Blueprint $table) {
                $table->bigIncrements('id');
                $table->unsignedBigInteger('invoice_id');
                $table->string('description');
                $table->decimal('qty', 10, 2)->default(1);
                $table->decimal('unit_price', 12, 2)->default(0);
                $table->decimal('line_total', 12, 2)->default(0);
                $table->timestamps();
                $table->index('invoice_id');
            });
        }
    }

    public function down(): void
    {
        Schema::dropIfExists('einvoice_invoice_items');
    }
};
