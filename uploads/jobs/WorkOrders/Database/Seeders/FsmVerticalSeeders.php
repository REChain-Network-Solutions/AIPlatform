<?php
namespace Modules\WorkOrders\Database\Seeders; use Illuminate\Database\Seeder;
use Modules\WorkOrders\Entities\{ChecklistTemplate,ChecklistItem,SlaProfile,CatalogItem,RateCard};
class FsmVerticalSeeders extends Seeder{
  public function run(): void {
    $sets=[
      'general_trades'=>[
        'templates'=>[['code'=>'gt_basic','label'=>'Basic Service']],
        'items'=>['gt_basic'=>['Site safety check','Confirm scope','Do work','QA review','Cleanup']],
        'sla'=>['arrival'=>480,'complete'=>1440],
        'catalog'=>[['sku'=>'GT-CALL','name'=>'Call-out','price'=>95,'unit'=>'ea']],
        'rate'=>['hourly'=>85,'callout'=>95]
      ],
      'hvac'=>[
        'templates'=>[['code'=>'hvac_maintenance','label'=>'HVAC Maintenance']],
        'items'=>['hvac_maintenance'=>['Check thermostat','Inspect filters','Clean condenser','Record pressures','Test run']],
        'sla'=>['arrival'=>240,'complete'=>1440],
        'catalog'=>[['sku'=>'HV-FLTR','name'=>'Filter (standard)','price'=>19.9,'unit'=>'ea']],
        'rate'=>['hourly'=>120,'callout'=>95]
      ],
    ];
    foreach($sets as $vertical=>$cfg){
      SlaProfile::updateOrCreate(['vertical'=>$vertical],[ 'arrival_minutes'=>$cfg['sla']['arrival'],'completion_minutes'=>$cfg['sla']['complete'] ]);
      RateCard::updateOrCreate(['vertical'=>$vertical,'code'=>'default'],['hourly'=>$cfg['rate']['hourly'],'callout'=>$cfg['rate']['callout']]);
      foreach($cfg['catalog'] as $it){
        CatalogItem::updateOrCreate(['vertical'=>$vertical,'name'=>$it['name']],['sku'=>$it['sku']??null,'price'=>$it['price'],'unit'=>$it['unit']??'ea']);
      }
      foreach($cfg['templates'] as $tpl){
        $t=ChecklistTemplate::firstOrCreate(['code'=>$tpl['code']],['label'=>$tpl['label'],'vertical'=>$vertical]); $o=1;
        foreach($cfg['items'][$tpl['code']] as $text){
          ChecklistItem::firstOrCreate(['template_id'=>$t->id,'order'=>$o],['text'=>$text,'required'=>true]); $o++;
        }
      }
    }
  }
}