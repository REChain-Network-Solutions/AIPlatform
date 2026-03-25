<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

return new class extends Migration {

    /**
     * Run the migrations.
     *
     * @return void
     */
    public function up()
    {
        if (!Schema::hasTable('e_invoice_company_settings')) {
            Schema::create('e_invoice_company_settings', function (Blueprint $table) {
                $table->increments('id');
                $table->integer('company_id')->unsigned()->nullable();
                $table->foreign('company_id')->references('id')->on('companies')->onDelete('cascade')->onUpdate('cascade');
                $table->string('electronic_address')->nullable();
                $table->string('electronic_address_scheme')->nullable();
                $table->string('e_invoice_company_id')->nullable();
                $table->string('e_invoice_company_id_scheme')->nullable();
                $table->timestamps();
            });
        }
    }

    /**
     * Reverse the migrations.
     *
     * @return void
     */
    public function down()
    {
        Schema::dropIfExists('e_invoice_company_settings');
    }

};
