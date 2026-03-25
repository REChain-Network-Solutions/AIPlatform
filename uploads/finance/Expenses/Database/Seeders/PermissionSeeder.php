<?php
namespace Modules\Expenses\Database\Seeders; use Illuminate\Database\Seeder; use Illuminate\Support\Facades\DB; use Illuminate\Support\Facades\Schema;
class PermissionSeeder extends Seeder{public function run():void{if(!Schema::hasTable('permissions'))return;DB::table('permissions')->updateOrInsert(['name':'expenses.access'],['title':'Expenses Access']);}}