<?php
namespace Modules\WorkOrders\Database\Seeders;
use Illuminate\Database\Seeder; use Illuminate\Support\Facades\DB; use Illuminate\Support\Facades\Schema;
class FieldServiceMenuSeeder extends Seeder {
  public function run(): void {
    $tables=['menus','menu_items','sidebar_menus']; $table=null;
    foreach($tables as $t){ if(Schema::hasTable($t)){ $table=$t; break; } }
    if(!$table){ $this->command?->warn('No menu table found.'); return; }
    $items=[
      ['label'=>'Work Orders','url'=>'/workorders','permission'=>'workorders.view','icon'=>'fa-clipboard-list','parent'=>null,'order'=>10],
      ['label'=>'Create Work Order','url'=>'/workorders/create','permission'=>'workorders.create','icon'=>null,'parent'=>'Work Orders','order'=>11],
      ['label'=>'Assignments','url'=>'/contractors/assignments','permission'=>'contractors.view_assignments','icon'=>null,'parent'=>null,'order'=>20],
      ['label'=>'Schedule (Day)','url'=>'/contractors/schedule','permission'=>'contractors.view_assignments','icon'=>null,'parent'=>'Assignments','order'=>21],
    ];
    foreach($items as $it){ DB::table($table)->updateOrInsert(['label'=>$it['label']], [
        'url'=>$it['url'],'permission'=>$it['permission'],'icon'=>$it['icon'],
        'parent_id'=>null,'order'=>$it['order'],'created_at'=>now(),'updated_at'=>now()
    ]); }
  }
}