<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;
use Modules\EInvoice\Entities\EInvoiceSetting;

return new class extends Migration {

    /**
     * Run the migrations.
     *
     * @return void
     */
    public function up()
    {
        \App\Models\Module::validateVersion(EInvoiceSetting::MODULE_NAME);

        if (!Schema::hasTable('e_invoice_settings')) {
            Schema::create('e_invoice_settings', function (Blueprint $table) {
                $table->id();
                $table->string('purchase_code')->nullable();
                $table->timestamp('supported_until')->nullable();
                $table->boolean('notify_update')->default(1);
                $table->timestamps();
            });

            EInvoiceSetting::create([]);
        }
    }

    /**
     * Reverse the migrations.
     *
     * @return void
     */
    public function down()
    {
        Schema::dropIfExists('e_invoice_settings');
    }

};
