# [YooKassa API SDK](../home.md)

# Class: \YooKassa\Model\Receipt\ReceiptItem
### Namespace: [\YooKassa\Model\Receipt](../namespaces/yookassa-model-receipt.md)
---
**Summary:**

Информация о товарной позиции в заказе. Используется для формирования чека.


---
### Constants
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| public | [DESCRIPTION_MAX_LENGTH](../classes/YooKassa-Model-Receipt-ReceiptItem.md#constant_DESCRIPTION_MAX_LENGTH) |  |  |
| public | [ADD_PROPS_MAX_LENGTH](../classes/YooKassa-Model-Receipt-ReceiptItem.md#constant_ADD_PROPS_MAX_LENGTH) |  |  |

---
### Properties
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| public | [$additional_payment_subject_props](../classes/YooKassa-Model-Receipt-ReceiptItem.md#property_additional_payment_subject_props) |  | Дополнительный реквизит предмета расчета (тег в 54 ФЗ — 1191) |
| public | [$additionalPaymentSubjectProps](../classes/YooKassa-Model-Receipt-ReceiptItem.md#property_additionalPaymentSubjectProps) |  | Дополнительный реквизит предмета расчета (тег в 54 ФЗ — 1191) |
| public | [$agent_type](../classes/YooKassa-Model-Receipt-ReceiptItem.md#property_agent_type) |  | Тип посредника, реализующего товар или услугу |
| public | [$agentType](../classes/YooKassa-Model-Receipt-ReceiptItem.md#property_agentType) |  | Тип посредника, реализующего товар или услугу |
| public | [$amount](../classes/YooKassa-Model-Receipt-ReceiptItem.md#property_amount) |  | Суммарная стоимость покупаемого товара в копейках/центах |
| public | [$country_of_origin_code](../classes/YooKassa-Model-Receipt-ReceiptItem.md#property_country_of_origin_code) |  | Код страны происхождения товара (тег в 54 ФЗ — 1230) |
| public | [$countryOfOriginCode](../classes/YooKassa-Model-Receipt-ReceiptItem.md#property_countryOfOriginCode) |  | Код страны происхождения товара (тег в 54 ФЗ — 1230) |
| public | [$customs_declaration_number](../classes/YooKassa-Model-Receipt-ReceiptItem.md#property_customs_declaration_number) |  | Номер таможенной декларации (от 1 до 32 символов). Тег в 54 ФЗ — 1231 |
| public | [$customsDeclarationNumber](../classes/YooKassa-Model-Receipt-ReceiptItem.md#property_customsDeclarationNumber) |  | Номер таможенной декларации (от 1 до 32 символов). Тег в 54 ФЗ — 1231 |
| public | [$description](../classes/YooKassa-Model-Receipt-ReceiptItem.md#property_description) |  | Наименование товара (тег в 54 ФЗ — 1030) |
| public | [$excise](../classes/YooKassa-Model-Receipt-ReceiptItem.md#property_excise) |  | Сумма акциза товара с учетом копеек (тег в 54 ФЗ — 1229) |
| public | [$is_shipping](../classes/YooKassa-Model-Receipt-ReceiptItem.md#property_is_shipping) |  | Флаг доставки |
| public | [$isShipping](../classes/YooKassa-Model-Receipt-ReceiptItem.md#property_isShipping) |  | Флаг доставки |
| public | [$mark_code_info](../classes/YooKassa-Model-Receipt-ReceiptItem.md#property_mark_code_info) |  | Код товара (тег в 54 ФЗ — 1163) |
| public | [$mark_mode](../classes/YooKassa-Model-Receipt-ReceiptItem.md#property_mark_mode) |  | Режим обработки кода маркировки (тег в 54 ФЗ — 2102) |
| public | [$mark_quantity](../classes/YooKassa-Model-Receipt-ReceiptItem.md#property_mark_quantity) |  | Дробное количество маркированного товара (тег в 54 ФЗ — 1291) |
| public | [$markCodeInfo](../classes/YooKassa-Model-Receipt-ReceiptItem.md#property_markCodeInfo) |  | Код товара (тег в 54 ФЗ — 1163) |
| public | [$markMode](../classes/YooKassa-Model-Receipt-ReceiptItem.md#property_markMode) |  | Режим обработки кода маркировки (тег в 54 ФЗ — 2102) |
| public | [$markQuantity](../classes/YooKassa-Model-Receipt-ReceiptItem.md#property_markQuantity) |  | Дробное количество маркированного товара (тег в 54 ФЗ — 1291) |
| public | [$measure](../classes/YooKassa-Model-Receipt-ReceiptItem.md#property_measure) |  | Мера количества предмета расчета (тег в 54 ФЗ — 2108) |
| public | [$payment_mode](../classes/YooKassa-Model-Receipt-ReceiptItem.md#property_payment_mode) |  | Признак способа расчета (тег в 54 ФЗ — 1214) |
| public | [$payment_subject](../classes/YooKassa-Model-Receipt-ReceiptItem.md#property_payment_subject) |  | Признак предмета расчета (тег в 54 ФЗ — 1212) |
| public | [$payment_subject_industry_details](../classes/YooKassa-Model-Receipt-ReceiptItem.md#property_payment_subject_industry_details) |  | Отраслевой реквизит предмета расчета (тег в 54 ФЗ — 1260) |
| public | [$paymentMode](../classes/YooKassa-Model-Receipt-ReceiptItem.md#property_paymentMode) |  | Признак способа расчета (тег в 54 ФЗ — 1214) |
| public | [$paymentSubject](../classes/YooKassa-Model-Receipt-ReceiptItem.md#property_paymentSubject) |  | Признак предмета расчета (тег в 54 ФЗ — 1212) |
| public | [$paymentSubjectIndustryDetails](../classes/YooKassa-Model-Receipt-ReceiptItem.md#property_paymentSubjectIndustryDetails) |  | Отраслевой реквизит предмета расчета (тег в 54 ФЗ — 1260) |
| public | [$price](../classes/YooKassa-Model-Receipt-ReceiptItem.md#property_price) |  | Цена товара (тег в 54 ФЗ — 1079) |
| public | [$product_code](../classes/YooKassa-Model-Receipt-ReceiptItem.md#property_product_code) |  | Код товара (тег в 54 ФЗ — 1162) |
| public | [$productCode](../classes/YooKassa-Model-Receipt-ReceiptItem.md#property_productCode) |  | Код товара (тег в 54 ФЗ — 1162) |
| public | [$quantity](../classes/YooKassa-Model-Receipt-ReceiptItem.md#property_quantity) |  | Количество (тег в 54 ФЗ — 1023) |
| public | [$supplier](../classes/YooKassa-Model-Receipt-ReceiptItem.md#property_supplier) |  | Информация о поставщике товара или услуги (тег в 54 ФЗ — 1224) |
| public | [$vat_code](../classes/YooKassa-Model-Receipt-ReceiptItem.md#property_vat_code) |  | Ставка НДС (тег в 54 ФЗ — 1199), число 1-10 |
| public | [$vatCode](../classes/YooKassa-Model-Receipt-ReceiptItem.md#property_vatCode) |  | Ставка НДС (тег в 54 ФЗ — 1199), число 1-10 |

---
### Methods
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| public | [__construct()](../classes/YooKassa-Common-AbstractObject.md#method___construct) |  | AbstractObject constructor. |
| public | [__get()](../classes/YooKassa-Common-AbstractObject.md#method___get) |  | Возвращает значение свойства. |
| public | [__isset()](../classes/YooKassa-Common-AbstractObject.md#method___isset) |  | Проверяет наличие свойства. |
| public | [__set()](../classes/YooKassa-Common-AbstractObject.md#method___set) |  | Устанавливает значение свойства. |
| public | [__unset()](../classes/YooKassa-Common-AbstractObject.md#method___unset) |  | Удаляет свойство. |
| public | [applyDiscountCoefficient()](../classes/YooKassa-Model-Receipt-ReceiptItem.md#method_applyDiscountCoefficient) |  | Применяет для товара скидку. |
| public | [fetchItem()](../classes/YooKassa-Model-Receipt-ReceiptItem.md#method_fetchItem) |  | Уменьшает количество покупаемого товара на указанное, возвращает объект позиции в чеке с уменьшаемым количеством |
| public | [fromArray()](../classes/YooKassa-Model-Receipt-ReceiptItem.md#method_fromArray) |  | Устанавливает значения свойств текущего объекта из массива. |
| public | [getAdditionalPaymentSubjectProps()](../classes/YooKassa-Model-Receipt-ReceiptItem.md#method_getAdditionalPaymentSubjectProps) |  | Возвращает дополнительный реквизит предмета расчета. |
| public | [getAgentType()](../classes/YooKassa-Model-Receipt-ReceiptItem.md#method_getAgentType) |  | Возвращает тип посредника, реализующего товар или услугу. |
| public | [getAmount()](../classes/YooKassa-Model-Receipt-ReceiptItem.md#method_getAmount) |  | Возвращает общую стоимость покупаемого товара в копейках/центах. |
| public | [getCountryOfOriginCode()](../classes/YooKassa-Model-Receipt-ReceiptItem.md#method_getCountryOfOriginCode) |  | Возвращает код страны происхождения товара по общероссийскому классификатору стран мира. |
| public | [getCustomsDeclarationNumber()](../classes/YooKassa-Model-Receipt-ReceiptItem.md#method_getCustomsDeclarationNumber) |  | Возвращает номер таможенной декларации. |
| public | [getDescription()](../classes/YooKassa-Model-Receipt-ReceiptItem.md#method_getDescription) |  | Возвращает наименование товара. |
| public | [getExcise()](../classes/YooKassa-Model-Receipt-ReceiptItem.md#method_getExcise) |  | Возвращает сумму акциза товара с учетом копеек. |
| public | [getMarkCodeInfo()](../classes/YooKassa-Model-Receipt-ReceiptItem.md#method_getMarkCodeInfo) |  | Возвращает код товара. |
| public | [getMarkMode()](../classes/YooKassa-Model-Receipt-ReceiptItem.md#method_getMarkMode) |  | Возвращает режим обработки кода маркировки. |
| public | [getMarkQuantity()](../classes/YooKassa-Model-Receipt-ReceiptItem.md#method_getMarkQuantity) |  | Возвращает дробное количество маркированного товара. |
| public | [getMeasure()](../classes/YooKassa-Model-Receipt-ReceiptItem.md#method_getMeasure) |  | Возвращает меру количества предмета расчета. |
| public | [getPaymentMode()](../classes/YooKassa-Model-Receipt-ReceiptItem.md#method_getPaymentMode) |  | Возвращает признак способа расчета. |
| public | [getPaymentSubject()](../classes/YooKassa-Model-Receipt-ReceiptItem.md#method_getPaymentSubject) |  | Возвращает признак предмета расчета. |
| public | [getPaymentSubjectIndustryDetails()](../classes/YooKassa-Model-Receipt-ReceiptItem.md#method_getPaymentSubjectIndustryDetails) |  | Возвращает отраслевой реквизит чека. |
| public | [getPrice()](../classes/YooKassa-Model-Receipt-ReceiptItem.md#method_getPrice) |  | Возвращает цену товара. |
| public | [getProductCode()](../classes/YooKassa-Model-Receipt-ReceiptItem.md#method_getProductCode) |  | Возвращает код товара — уникальный номер, который присваивается экземпляру товара при маркировке. |
| public | [getQuantity()](../classes/YooKassa-Model-Receipt-ReceiptItem.md#method_getQuantity) |  | Возвращает количество товара. |
| public | [getSupplier()](../classes/YooKassa-Model-Receipt-ReceiptItem.md#method_getSupplier) |  | Возвращает информацию о поставщике товара или услуги. |
| public | [getValidator()](../classes/YooKassa-Common-AbstractObject.md#method_getValidator) |  |  |
| public | [getVatCode()](../classes/YooKassa-Model-Receipt-ReceiptItem.md#method_getVatCode) |  | Возвращает ставку НДС |
| public | [increasePrice()](../classes/YooKassa-Model-Receipt-ReceiptItem.md#method_increasePrice) |  | Увеличивает цену товара на указанную величину. |
| public | [isShipping()](../classes/YooKassa-Model-Receipt-ReceiptItem.md#method_isShipping) |  | Проверяет, является ли текущий элемент чека доставкой. |
| public | [jsonSerialize()](../classes/YooKassa-Model-Receipt-ReceiptItem.md#method_jsonSerialize) |  | Возвращает ассоциативный массив со свойствами текущего объекта для его дальнейшей JSON сериализации. |
| public | [offsetExists()](../classes/YooKassa-Common-AbstractObject.md#method_offsetExists) |  | Проверяет наличие свойства. |
| public | [offsetGet()](../classes/YooKassa-Common-AbstractObject.md#method_offsetGet) |  | Возвращает значение свойства. |
| public | [offsetSet()](../classes/YooKassa-Common-AbstractObject.md#method_offsetSet) |  | Устанавливает значение свойства. |
| public | [offsetUnset()](../classes/YooKassa-Common-AbstractObject.md#method_offsetUnset) |  | Удаляет свойство. |
| public | [setAdditionalPaymentSubjectProps()](../classes/YooKassa-Model-Receipt-ReceiptItem.md#method_setAdditionalPaymentSubjectProps) |  | Устанавливает дополнительный реквизит предмета расчета. |
| public | [setAgentType()](../classes/YooKassa-Model-Receipt-ReceiptItem.md#method_setAgentType) |  | Устанавливает тип посредника, реализующего товар или услугу. |
| public | [setCountryOfOriginCode()](../classes/YooKassa-Model-Receipt-ReceiptItem.md#method_setCountryOfOriginCode) |  | Устанавливает код страны происхождения товара по общероссийскому классификатору стран мира. |
| public | [setCustomsDeclarationNumber()](../classes/YooKassa-Model-Receipt-ReceiptItem.md#method_setCustomsDeclarationNumber) |  | Устанавливает номер таможенной декларации (от 1 до 32 символов). |
| public | [setDescription()](../classes/YooKassa-Model-Receipt-ReceiptItem.md#method_setDescription) |  | Устанавливает наименование товара. |
| public | [setExcise()](../classes/YooKassa-Model-Receipt-ReceiptItem.md#method_setExcise) |  | Устанавливает сумму акциза товара с учетом копеек. |
| public | [setIsShipping()](../classes/YooKassa-Model-Receipt-ReceiptItem.md#method_setIsShipping) |  | Устанавливает флаг доставки для текущего объекта айтема в чеке. |
| public | [setMarkCodeInfo()](../classes/YooKassa-Model-Receipt-ReceiptItem.md#method_setMarkCodeInfo) |  | Устанавливает код товара. |
| public | [setMarkMode()](../classes/YooKassa-Model-Receipt-ReceiptItem.md#method_setMarkMode) |  | Устанавливает режим обработки кода маркировки. |
| public | [setMarkQuantity()](../classes/YooKassa-Model-Receipt-ReceiptItem.md#method_setMarkQuantity) |  | Устанавливает дробное количество маркированного товара. |
| public | [setMeasure()](../classes/YooKassa-Model-Receipt-ReceiptItem.md#method_setMeasure) |  | Устанавливает меру количества предмета расчета. |
| public | [setPaymentMode()](../classes/YooKassa-Model-Receipt-ReceiptItem.md#method_setPaymentMode) |  | Устанавливает признак способа расчета. |
| public | [setPaymentSubject()](../classes/YooKassa-Model-Receipt-ReceiptItem.md#method_setPaymentSubject) |  | Устанавливает признак предмета расчета. |
| public | [setPaymentSubjectIndustryDetails()](../classes/YooKassa-Model-Receipt-ReceiptItem.md#method_setPaymentSubjectIndustryDetails) |  | Устанавливает отраслевой реквизит чека. |
| public | [setPrice()](../classes/YooKassa-Model-Receipt-ReceiptItem.md#method_setPrice) |  | Устанавливает цену товара. |
| public | [setProductCode()](../classes/YooKassa-Model-Receipt-ReceiptItem.md#method_setProductCode) |  | Устанавливает код товара — уникальный номер, который присваивается экземпляру товара при маркировке. |
| public | [setQuantity()](../classes/YooKassa-Model-Receipt-ReceiptItem.md#method_setQuantity) |  | Устанавливает количество покупаемого товара. |
| public | [setSupplier()](../classes/YooKassa-Model-Receipt-ReceiptItem.md#method_setSupplier) |  | Устанавливает информацию о поставщике товара или услуги. |
| public | [setVatCode()](../classes/YooKassa-Model-Receipt-ReceiptItem.md#method_setVatCode) |  | Устанавливает ставку НДС |
| public | [toArray()](../classes/YooKassa-Common-AbstractObject.md#method_toArray) |  | Возвращает ассоциативный массив со свойствами текущего объекта для его дальнейшей JSON сериализации Является алиасом метода AbstractObject::jsonSerialize(). |
| protected | [getUnknownProperties()](../classes/YooKassa-Common-AbstractObject.md#method_getUnknownProperties) |  | Возвращает массив свойств которые не существуют, но были заданы у объекта. |
| protected | [validatePropertyValue()](../classes/YooKassa-Common-AbstractObject.md#method_validatePropertyValue) |  |  |

---
### Details
* File: [lib/Model/Receipt/ReceiptItem.php](../../lib/Model/Receipt/ReceiptItem.php)
* Package: YooKassa\Model
* Class Hierarchy: 
  * [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)
  * \YooKassa\Model\Receipt\ReceiptItem
* Implements:
  * [\YooKassa\Model\Receipt\ReceiptItemInterface](../classes/YooKassa-Model-Receipt-ReceiptItemInterface.md)

* See Also:
  * [](https://yookassa.ru/developers/api)

---
### Tags
| Tag | Version | Description |
| --- | ------- | ----------- |
| category |  | Class |
| author |  | cms@yoomoney.ru |

---
## Constants
<a name="constant_DESCRIPTION_MAX_LENGTH" class="anchor"></a>
###### DESCRIPTION_MAX_LENGTH
```php
DESCRIPTION_MAX_LENGTH = 128 : int
```


<a name="constant_ADD_PROPS_MAX_LENGTH" class="anchor"></a>
###### ADD_PROPS_MAX_LENGTH
```php
ADD_PROPS_MAX_LENGTH = 64 : int
```



---
## Properties
<a name="property_additional_payment_subject_props"></a>
#### public $additional_payment_subject_props : string
---
***Description***

Дополнительный реквизит предмета расчета (тег в 54 ФЗ — 1191)

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**


<a name="property_additionalPaymentSubjectProps"></a>
#### public $additionalPaymentSubjectProps : string
---
***Description***

Дополнительный реквизит предмета расчета (тег в 54 ФЗ — 1191)

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**


<a name="property_agent_type"></a>
#### public $agent_type : string
---
***Description***

Тип посредника, реализующего товар или услугу

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**


<a name="property_agentType"></a>
#### public $agentType : string
---
***Description***

Тип посредника, реализующего товар или услугу

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**


<a name="property_amount"></a>
#### public $amount : float
---
***Description***

Суммарная стоимость покупаемого товара в копейках/центах

**Type:** <a href="../float"><abbr title="float">float</abbr></a>

**Details:**


<a name="property_country_of_origin_code"></a>
#### public $country_of_origin_code : string
---
***Description***

Код страны происхождения товара (тег в 54 ФЗ — 1230)

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**


<a name="property_countryOfOriginCode"></a>
#### public $countryOfOriginCode : string
---
***Description***

Код страны происхождения товара (тег в 54 ФЗ — 1230)

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**


<a name="property_customs_declaration_number"></a>
#### public $customs_declaration_number : string
---
***Description***

Номер таможенной декларации (от 1 до 32 символов). Тег в 54 ФЗ — 1231

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**


<a name="property_customsDeclarationNumber"></a>
#### public $customsDeclarationNumber : string
---
***Description***

Номер таможенной декларации (от 1 до 32 символов). Тег в 54 ФЗ — 1231

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**


<a name="property_description"></a>
#### public $description : string
---
***Description***

Наименование товара (тег в 54 ФЗ — 1030)

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**


<a name="property_excise"></a>
#### public $excise : float
---
***Description***

Сумма акциза товара с учетом копеек (тег в 54 ФЗ — 1229)

**Type:** <a href="../float"><abbr title="float">float</abbr></a>

**Details:**


<a name="property_is_shipping"></a>
#### public $is_shipping : bool
---
***Description***

Флаг доставки

**Type:** <a href="../bool"><abbr title="bool">bool</abbr></a>

**Details:**


<a name="property_isShipping"></a>
#### public $isShipping : bool
---
***Description***

Флаг доставки

**Type:** <a href="../bool"><abbr title="bool">bool</abbr></a>

**Details:**


<a name="property_mark_code_info"></a>
#### public $mark_code_info : \YooKassa\Model\Receipt\MarkCodeInfo
---
***Description***

Код товара (тег в 54 ФЗ — 1163)

**Type:** <a href="../classes/YooKassa-Model-Receipt-MarkCodeInfo.html"><abbr title="\YooKassa\Model\Receipt\MarkCodeInfo">MarkCodeInfo</abbr></a>

**Details:**


<a name="property_mark_mode"></a>
#### public $mark_mode : string
---
***Description***

Режим обработки кода маркировки (тег в 54 ФЗ — 2102)

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**


<a name="property_mark_quantity"></a>
#### public $mark_quantity : \YooKassa\Model\Receipt\MarkQuantity
---
***Description***

Дробное количество маркированного товара (тег в 54 ФЗ — 1291)

**Type:** <a href="../classes/YooKassa-Model-Receipt-MarkQuantity.html"><abbr title="\YooKassa\Model\Receipt\MarkQuantity">MarkQuantity</abbr></a>

**Details:**


<a name="property_markCodeInfo"></a>
#### public $markCodeInfo : \YooKassa\Model\Receipt\MarkCodeInfo
---
***Description***

Код товара (тег в 54 ФЗ — 1163)

**Type:** <a href="../classes/YooKassa-Model-Receipt-MarkCodeInfo.html"><abbr title="\YooKassa\Model\Receipt\MarkCodeInfo">MarkCodeInfo</abbr></a>

**Details:**


<a name="property_markMode"></a>
#### public $markMode : string
---
***Description***

Режим обработки кода маркировки (тег в 54 ФЗ — 2102)

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**


<a name="property_markQuantity"></a>
#### public $markQuantity : \YooKassa\Model\Receipt\MarkQuantity
---
***Description***

Дробное количество маркированного товара (тег в 54 ФЗ — 1291)

**Type:** <a href="../classes/YooKassa-Model-Receipt-MarkQuantity.html"><abbr title="\YooKassa\Model\Receipt\MarkQuantity">MarkQuantity</abbr></a>

**Details:**


<a name="property_measure"></a>
#### public $measure : string
---
***Description***

Мера количества предмета расчета (тег в 54 ФЗ — 2108)

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**


<a name="property_payment_mode"></a>
#### public $payment_mode : string
---
***Description***

Признак способа расчета (тег в 54 ФЗ — 1214)

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**


<a name="property_payment_subject"></a>
#### public $payment_subject : string
---
***Description***

Признак предмета расчета (тег в 54 ФЗ — 1212)

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**


<a name="property_payment_subject_industry_details"></a>
#### public $payment_subject_industry_details : \YooKassa\Common\ListObjectInterface|\YooKassa\Model\Receipt\IndustryDetails[]
---
***Description***

Отраслевой реквизит предмета расчета (тег в 54 ФЗ — 1260)

**Type:** <a href="../\YooKassa\Common\ListObjectInterface|\YooKassa\Model\Receipt\IndustryDetails[]"><abbr title="\YooKassa\Common\ListObjectInterface|\YooKassa\Model\Receipt\IndustryDetails[]">IndustryDetails[]</abbr></a>

**Details:**


<a name="property_paymentMode"></a>
#### public $paymentMode : string
---
***Description***

Признак способа расчета (тег в 54 ФЗ — 1214)

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**


<a name="property_paymentSubject"></a>
#### public $paymentSubject : string
---
***Description***

Признак предмета расчета (тег в 54 ФЗ — 1212)

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**


<a name="property_paymentSubjectIndustryDetails"></a>
#### public $paymentSubjectIndustryDetails : \YooKassa\Common\ListObjectInterface|\YooKassa\Model\Receipt\IndustryDetails[]
---
***Description***

Отраслевой реквизит предмета расчета (тег в 54 ФЗ — 1260)

**Type:** <a href="../\YooKassa\Common\ListObjectInterface|\YooKassa\Model\Receipt\IndustryDetails[]"><abbr title="\YooKassa\Common\ListObjectInterface|\YooKassa\Model\Receipt\IndustryDetails[]">IndustryDetails[]</abbr></a>

**Details:**


<a name="property_price"></a>
#### public $price : \YooKassa\Model\AmountInterface
---
***Description***

Цена товара (тег в 54 ФЗ — 1079)

**Type:** <a href="../classes/YooKassa-Model-AmountInterface.html"><abbr title="\YooKassa\Model\AmountInterface">AmountInterface</abbr></a>

**Details:**


<a name="property_product_code"></a>
#### public $product_code : string
---
***Description***

Код товара (тег в 54 ФЗ — 1162)

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**


<a name="property_productCode"></a>
#### public $productCode : string
---
***Description***

Код товара (тег в 54 ФЗ — 1162)

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**


<a name="property_quantity"></a>
#### public $quantity : float
---
***Description***

Количество (тег в 54 ФЗ — 1023)

**Type:** <a href="../float"><abbr title="float">float</abbr></a>

**Details:**


<a name="property_supplier"></a>
#### public $supplier : \YooKassa\Model\Receipt\Supplier
---
***Description***

Информация о поставщике товара или услуги (тег в 54 ФЗ — 1224)

**Type:** <a href="../classes/YooKassa-Model-Receipt-Supplier.html"><abbr title="\YooKassa\Model\Receipt\Supplier">Supplier</abbr></a>

**Details:**


<a name="property_vat_code"></a>
#### public $vat_code : int
---
***Description***

Ставка НДС (тег в 54 ФЗ — 1199), число 1-10

**Type:** <a href="../int"><abbr title="int">int</abbr></a>

**Details:**


<a name="property_vatCode"></a>
#### public $vatCode : int
---
***Description***

Ставка НДС (тег в 54 ФЗ — 1199), число 1-10

**Type:** <a href="../int"><abbr title="int">int</abbr></a>

**Details:**



---
## Methods
<a name="method___construct" class="anchor"></a>
#### public __construct() : mixed

```php
public __construct(array|null $data = []) : mixed
```

**Summary**

AbstractObject constructor.

**Details:**
* Inherited From: [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">array OR null</code> | data  |  |

**Returns:** mixed - 


<a name="method___get" class="anchor"></a>
#### public __get() : mixed

```php
public __get(string $propertyName) : mixed
```

**Summary**

Возвращает значение свойства.

**Details:**
* Inherited From: [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string</code> | propertyName  | Имя свойства |

**Returns:** mixed - Значение свойства


<a name="method___isset" class="anchor"></a>
#### public __isset() : bool

```php
public __isset(string $propertyName) : bool
```

**Summary**

Проверяет наличие свойства.

**Details:**
* Inherited From: [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string</code> | propertyName  | Имя проверяемого свойства |

**Returns:** bool - True если свойство имеется, false если нет


<a name="method___set" class="anchor"></a>
#### public __set() : void

```php
public __set(string $propertyName, mixed $value) : void
```

**Summary**

Устанавливает значение свойства.

**Details:**
* Inherited From: [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string</code> | propertyName  | Имя свойства |
| <code lang="php">mixed</code> | value  | Значение свойства |

**Returns:** void - 


<a name="method___unset" class="anchor"></a>
#### public __unset() : void

```php
public __unset(string $propertyName) : void
```

**Summary**

Удаляет свойство.

**Details:**
* Inherited From: [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string</code> | propertyName  | Имя удаляемого свойства |

**Returns:** void - 


<a name="method_applyDiscountCoefficient" class="anchor"></a>
#### public applyDiscountCoefficient() : void

```php
public applyDiscountCoefficient(float|null $coefficient) : void
```

**Summary**

Применяет для товара скидку.

**Details:**
* Inherited From: [\YooKassa\Model\Receipt\ReceiptItem](../classes/YooKassa-Model-Receipt-ReceiptItem.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">float OR null</code> | coefficient  | Множитель скидки |

**Returns:** void - 


<a name="method_fetchItem" class="anchor"></a>
#### public fetchItem() : \YooKassa\Model\Receipt\ReceiptItem

```php
public fetchItem(float|null $count) : \YooKassa\Model\Receipt\ReceiptItem
```

**Summary**

Уменьшает количество покупаемого товара на указанное, возвращает объект позиции в чеке с уменьшаемым количеством

**Details:**
* Inherited From: [\YooKassa\Model\Receipt\ReceiptItem](../classes/YooKassa-Model-Receipt-ReceiptItem.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">float OR null</code> | count  | Количество на которое уменьшаем позицию в чеке |

##### Throws:
| Type | Description |
| ---- | ----------- |
| \YooKassa\Validator\Exceptions\EmptyPropertyValueException | Выбрасывается если было передано пустое значение |
| \YooKassa\Validator\Exceptions\InvalidPropertyValueException | Выбрасывается если в качестве аргумента был передан ноль или отрицательное число, или число больше текущего количества покупаемого товара |
| \YooKassa\Validator\Exceptions\InvalidPropertyValueTypeException | Выбрасывается если в качестве аргумента было передано не число |

**Returns:** \YooKassa\Model\Receipt\ReceiptItem - 


<a name="method_fromArray" class="anchor"></a>
#### public fromArray() : void

```php
public fromArray(array|\Traversable $sourceArray) : void
```

**Summary**

Устанавливает значения свойств текущего объекта из массива.

**Details:**
* Inherited From: [\YooKassa\Model\Receipt\ReceiptItem](../classes/YooKassa-Model-Receipt-ReceiptItem.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">array OR \Traversable</code> | sourceArray  | Ассоциативный массив с настройками |

**Returns:** void - 


<a name="method_getAdditionalPaymentSubjectProps" class="anchor"></a>
#### public getAdditionalPaymentSubjectProps() : string|null

```php
public getAdditionalPaymentSubjectProps() : string|null
```

**Summary**

Возвращает дополнительный реквизит предмета расчета.

**Details:**
* Inherited From: [\YooKassa\Model\Receipt\ReceiptItem](../classes/YooKassa-Model-Receipt-ReceiptItem.md)

**Returns:** string|null - Дополнительный реквизит предмета расчета


<a name="method_getAgentType" class="anchor"></a>
#### public getAgentType() : string|null

```php
public getAgentType() : string|null
```

**Summary**

Возвращает тип посредника, реализующего товар или услугу.

**Details:**
* Inherited From: [\YooKassa\Model\Receipt\ReceiptItem](../classes/YooKassa-Model-Receipt-ReceiptItem.md)

**Returns:** string|null - Тип посредника


<a name="method_getAmount" class="anchor"></a>
#### public getAmount() : int

```php
public getAmount() : int
```

**Summary**

Возвращает общую стоимость покупаемого товара в копейках/центах.

**Details:**
* Inherited From: [\YooKassa\Model\Receipt\ReceiptItem](../classes/YooKassa-Model-Receipt-ReceiptItem.md)

**Returns:** int - Сумма стоимости покупаемого товара


<a name="method_getCountryOfOriginCode" class="anchor"></a>
#### public getCountryOfOriginCode() : null|string

```php
public getCountryOfOriginCode() : null|string
```

**Summary**

Возвращает код страны происхождения товара по общероссийскому классификатору стран мира.

**Details:**
* Inherited From: [\YooKassa\Model\Receipt\ReceiptItem](../classes/YooKassa-Model-Receipt-ReceiptItem.md)

**Returns:** null|string - Код страны происхождения товара


<a name="method_getCustomsDeclarationNumber" class="anchor"></a>
#### public getCustomsDeclarationNumber() : null|string

```php
public getCustomsDeclarationNumber() : null|string
```

**Summary**

Возвращает номер таможенной декларации.

**Details:**
* Inherited From: [\YooKassa\Model\Receipt\ReceiptItem](../classes/YooKassa-Model-Receipt-ReceiptItem.md)

**Returns:** null|string - Номер таможенной декларации (от 1 до 32 символов)


<a name="method_getDescription" class="anchor"></a>
#### public getDescription() : string|null

```php
public getDescription() : string|null
```

**Summary**

Возвращает наименование товара.

**Details:**
* Inherited From: [\YooKassa\Model\Receipt\ReceiptItem](../classes/YooKassa-Model-Receipt-ReceiptItem.md)

**Returns:** string|null - Наименование товара


<a name="method_getExcise" class="anchor"></a>
#### public getExcise() : null|float

```php
public getExcise() : null|float
```

**Summary**

Возвращает сумму акциза товара с учетом копеек.

**Details:**
* Inherited From: [\YooKassa\Model\Receipt\ReceiptItem](../classes/YooKassa-Model-Receipt-ReceiptItem.md)

**Returns:** null|float - Сумма акциза товара с учетом копеек


<a name="method_getMarkCodeInfo" class="anchor"></a>
#### public getMarkCodeInfo() : \YooKassa\Model\Receipt\MarkCodeInfo|null

```php
public getMarkCodeInfo() : \YooKassa\Model\Receipt\MarkCodeInfo|null
```

**Summary**

Возвращает код товара.

**Details:**
* Inherited From: [\YooKassa\Model\Receipt\ReceiptItem](../classes/YooKassa-Model-Receipt-ReceiptItem.md)

**Returns:** \YooKassa\Model\Receipt\MarkCodeInfo|null - Код товара


<a name="method_getMarkMode" class="anchor"></a>
#### public getMarkMode() : string|null

```php
public getMarkMode() : string|null
```

**Summary**

Возвращает режим обработки кода маркировки.

**Details:**
* Inherited From: [\YooKassa\Model\Receipt\ReceiptItem](../classes/YooKassa-Model-Receipt-ReceiptItem.md)

**Returns:** string|null - Режим обработки кода маркировки


<a name="method_getMarkQuantity" class="anchor"></a>
#### public getMarkQuantity() : \YooKassa\Model\Receipt\MarkQuantity|null

```php
public getMarkQuantity() : \YooKassa\Model\Receipt\MarkQuantity|null
```

**Summary**

Возвращает дробное количество маркированного товара.

**Details:**
* Inherited From: [\YooKassa\Model\Receipt\ReceiptItem](../classes/YooKassa-Model-Receipt-ReceiptItem.md)

**Returns:** \YooKassa\Model\Receipt\MarkQuantity|null - Дробное количество маркированного товара


<a name="method_getMeasure" class="anchor"></a>
#### public getMeasure() : string|null

```php
public getMeasure() : string|null
```

**Summary**

Возвращает меру количества предмета расчета.

**Details:**
* Inherited From: [\YooKassa\Model\Receipt\ReceiptItem](../classes/YooKassa-Model-Receipt-ReceiptItem.md)

**Returns:** string|null - Мера количества предмета расчета


<a name="method_getPaymentMode" class="anchor"></a>
#### public getPaymentMode() : null|string

```php
public getPaymentMode() : null|string
```

**Summary**

Возвращает признак способа расчета.

**Details:**
* Inherited From: [\YooKassa\Model\Receipt\ReceiptItem](../classes/YooKassa-Model-Receipt-ReceiptItem.md)

**Returns:** null|string - Признак способа расчета


<a name="method_getPaymentSubject" class="anchor"></a>
#### public getPaymentSubject() : null|string

```php
public getPaymentSubject() : null|string
```

**Summary**

Возвращает признак предмета расчета.

**Details:**
* Inherited From: [\YooKassa\Model\Receipt\ReceiptItem](../classes/YooKassa-Model-Receipt-ReceiptItem.md)

**Returns:** null|string - Признак предмета расчета


<a name="method_getPaymentSubjectIndustryDetails" class="anchor"></a>
#### public getPaymentSubjectIndustryDetails() : \YooKassa\Model\Receipt\IndustryDetails[]|\YooKassa\Common\ListObjectInterface

```php
public getPaymentSubjectIndustryDetails() : \YooKassa\Model\Receipt\IndustryDetails[]|\YooKassa\Common\ListObjectInterface
```

**Summary**

Возвращает отраслевой реквизит чека.

**Details:**
* Inherited From: [\YooKassa\Model\Receipt\ReceiptItem](../classes/YooKassa-Model-Receipt-ReceiptItem.md)

**Returns:** \YooKassa\Model\Receipt\IndustryDetails[]|\YooKassa\Common\ListObjectInterface - Отраслевой реквизит чека


<a name="method_getPrice" class="anchor"></a>
#### public getPrice() : \YooKassa\Model\AmountInterface

```php
public getPrice() : \YooKassa\Model\AmountInterface
```

**Summary**

Возвращает цену товара.

**Details:**
* Inherited From: [\YooKassa\Model\Receipt\ReceiptItem](../classes/YooKassa-Model-Receipt-ReceiptItem.md)

**Returns:** \YooKassa\Model\AmountInterface - Цена товара


<a name="method_getProductCode" class="anchor"></a>
#### public getProductCode() : null|string

```php
public getProductCode() : null|string
```

**Summary**

Возвращает код товара — уникальный номер, который присваивается экземпляру товара при маркировке.

**Details:**
* Inherited From: [\YooKassa\Model\Receipt\ReceiptItem](../classes/YooKassa-Model-Receipt-ReceiptItem.md)

**Returns:** null|string - Код товара


<a name="method_getQuantity" class="anchor"></a>
#### public getQuantity() : float|null

```php
public getQuantity() : float|null
```

**Summary**

Возвращает количество товара.

**Details:**
* Inherited From: [\YooKassa\Model\Receipt\ReceiptItem](../classes/YooKassa-Model-Receipt-ReceiptItem.md)

**Returns:** float|null - Количество купленного товара


<a name="method_getSupplier" class="anchor"></a>
#### public getSupplier() : \YooKassa\Model\Receipt\Supplier|null

```php
public getSupplier() : \YooKassa\Model\Receipt\Supplier|null
```

**Summary**

Возвращает информацию о поставщике товара или услуги.

**Details:**
* Inherited From: [\YooKassa\Model\Receipt\ReceiptItem](../classes/YooKassa-Model-Receipt-ReceiptItem.md)

**Returns:** \YooKassa\Model\Receipt\Supplier|null - 


<a name="method_getValidator" class="anchor"></a>
#### public getValidator() : \YooKassa\Validator\Validator

```php
public getValidator() : \YooKassa\Validator\Validator
```

**Details:**
* Inherited From: [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)

**Returns:** \YooKassa\Validator\Validator - 


<a name="method_getVatCode" class="anchor"></a>
#### public getVatCode() : null|int

```php
public getVatCode() : null|int
```

**Summary**

Возвращает ставку НДС

**Details:**
* Inherited From: [\YooKassa\Model\Receipt\ReceiptItem](../classes/YooKassa-Model-Receipt-ReceiptItem.md)

**Returns:** null|int - Ставка НДС, число 1-10, или null, если ставка не задана


<a name="method_increasePrice" class="anchor"></a>
#### public increasePrice() : void

```php
public increasePrice(float|null $value) : void
```

**Summary**

Увеличивает цену товара на указанную величину.

**Details:**
* Inherited From: [\YooKassa\Model\Receipt\ReceiptItem](../classes/YooKassa-Model-Receipt-ReceiptItem.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">float OR null</code> | value  | Сумма на которую цену товара увеличиваем |

**Returns:** void - 


<a name="method_isShipping" class="anchor"></a>
#### public isShipping() : bool

```php
public isShipping() : bool
```

**Summary**

Проверяет, является ли текущий элемент чека доставкой.

**Details:**
* Inherited From: [\YooKassa\Model\Receipt\ReceiptItem](../classes/YooKassa-Model-Receipt-ReceiptItem.md)

**Returns:** bool - True если доставка, false если обычный товар


<a name="method_jsonSerialize" class="anchor"></a>
#### public jsonSerialize() : array

```php
public jsonSerialize() : array
```

**Summary**

Возвращает ассоциативный массив со свойствами текущего объекта для его дальнейшей JSON сериализации.

**Details:**
* Inherited From: [\YooKassa\Model\Receipt\ReceiptItem](../classes/YooKassa-Model-Receipt-ReceiptItem.md)

**Returns:** array - Ассоциативный массив со свойствами текущего объекта


<a name="method_offsetExists" class="anchor"></a>
#### public offsetExists() : bool

```php
public offsetExists(string $offset) : bool
```

**Summary**

Проверяет наличие свойства.

**Details:**
* Inherited From: [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string</code> | offset  | Имя проверяемого свойства |

**Returns:** bool - True если свойство имеется, false если нет


<a name="method_offsetGet" class="anchor"></a>
#### public offsetGet() : mixed

```php
public offsetGet(string $offset) : mixed
```

**Summary**

Возвращает значение свойства.

**Details:**
* Inherited From: [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string</code> | offset  | Имя свойства |

**Returns:** mixed - Значение свойства


<a name="method_offsetSet" class="anchor"></a>
#### public offsetSet() : void

```php
public offsetSet(string $offset, mixed $value) : void
```

**Summary**

Устанавливает значение свойства.

**Details:**
* Inherited From: [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string</code> | offset  | Имя свойства |
| <code lang="php">mixed</code> | value  | Значение свойства |

**Returns:** void - 


<a name="method_offsetUnset" class="anchor"></a>
#### public offsetUnset() : void

```php
public offsetUnset(string $offset) : void
```

**Summary**

Удаляет свойство.

**Details:**
* Inherited From: [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string</code> | offset  | Имя удаляемого свойства |

**Returns:** void - 


<a name="method_setAdditionalPaymentSubjectProps" class="anchor"></a>
#### public setAdditionalPaymentSubjectProps() : void

```php
public setAdditionalPaymentSubjectProps(string|null $additional_payment_subject_props) : void
```

**Summary**

Устанавливает дополнительный реквизит предмета расчета.

**Details:**
* Inherited From: [\YooKassa\Model\Receipt\ReceiptItem](../classes/YooKassa-Model-Receipt-ReceiptItem.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | additional_payment_subject_props  | Дополнительный реквизит предмета расчета |

**Returns:** void - 


<a name="method_setAgentType" class="anchor"></a>
#### public setAgentType() : self

```php
public setAgentType(string|null $agent_type = null) : self
```

**Summary**

Устанавливает тип посредника, реализующего товар или услугу.

**Details:**
* Inherited From: [\YooKassa\Model\Receipt\ReceiptItem](../classes/YooKassa-Model-Receipt-ReceiptItem.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | agent_type  | Тип посредника |

**Returns:** self - 


<a name="method_setCountryOfOriginCode" class="anchor"></a>
#### public setCountryOfOriginCode() : self

```php
public setCountryOfOriginCode(string|null $country_of_origin_code = null) : self
```

**Summary**

Устанавливает код страны происхождения товара по общероссийскому классификатору стран мира.

**Details:**
* Inherited From: [\YooKassa\Model\Receipt\ReceiptItem](../classes/YooKassa-Model-Receipt-ReceiptItem.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | country_of_origin_code  | Код страны происхождения товара |

**Returns:** self - 


<a name="method_setCustomsDeclarationNumber" class="anchor"></a>
#### public setCustomsDeclarationNumber() : self

```php
public setCustomsDeclarationNumber(string|null $customs_declaration_number = null) : self
```

**Summary**

Устанавливает номер таможенной декларации (от 1 до 32 символов).

**Details:**
* Inherited From: [\YooKassa\Model\Receipt\ReceiptItem](../classes/YooKassa-Model-Receipt-ReceiptItem.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | customs_declaration_number  | Номер таможенной декларации |

**Returns:** self - 


<a name="method_setDescription" class="anchor"></a>
#### public setDescription() : self

```php
public setDescription(string|null $description = null) : self
```

**Summary**

Устанавливает наименование товара.

**Details:**
* Inherited From: [\YooKassa\Model\Receipt\ReceiptItem](../classes/YooKassa-Model-Receipt-ReceiptItem.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | description  | Наименование товара |

**Returns:** self - 


<a name="method_setExcise" class="anchor"></a>
#### public setExcise() : self

```php
public setExcise(float|null $excise = null) : self
```

**Summary**

Устанавливает сумму акциза товара с учетом копеек.

**Details:**
* Inherited From: [\YooKassa\Model\Receipt\ReceiptItem](../classes/YooKassa-Model-Receipt-ReceiptItem.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">float OR null</code> | excise  | Сумма акциза товара с учетом копеек |

**Returns:** self - 


<a name="method_setIsShipping" class="anchor"></a>
#### public setIsShipping() : self

```php
public setIsShipping(bool $value) : self
```

**Summary**

Устанавливает флаг доставки для текущего объекта айтема в чеке.

**Details:**
* Inherited From: [\YooKassa\Model\Receipt\ReceiptItem](../classes/YooKassa-Model-Receipt-ReceiptItem.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">bool</code> | value  | True если айтем является доставкой, false если нет |

**Returns:** self - 


<a name="method_setMarkCodeInfo" class="anchor"></a>
#### public setMarkCodeInfo() : self

```php
public setMarkCodeInfo(array|\YooKassa\Model\Receipt\MarkCodeInfo|null $mark_code_info = null) : self
```

**Summary**

Устанавливает код товара.

**Details:**
* Inherited From: [\YooKassa\Model\Receipt\ReceiptItem](../classes/YooKassa-Model-Receipt-ReceiptItem.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">array OR \YooKassa\Model\Receipt\MarkCodeInfo OR null</code> | mark_code_info  | Код товара |

**Returns:** self - 


<a name="method_setMarkMode" class="anchor"></a>
#### public setMarkMode() : self

```php
public setMarkMode(string|null $mark_mode = null) : self
```

**Summary**

Устанавливает режим обработки кода маркировки.

**Details:**
* Inherited From: [\YooKassa\Model\Receipt\ReceiptItem](../classes/YooKassa-Model-Receipt-ReceiptItem.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | mark_mode  | Режим обработки кода маркировки |

**Returns:** self - 


<a name="method_setMarkQuantity" class="anchor"></a>
#### public setMarkQuantity() : self

```php
public setMarkQuantity(array|\YooKassa\Model\Receipt\MarkQuantity|null $mark_quantity = null) : self
```

**Summary**

Устанавливает дробное количество маркированного товара.

**Details:**
* Inherited From: [\YooKassa\Model\Receipt\ReceiptItem](../classes/YooKassa-Model-Receipt-ReceiptItem.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">array OR \YooKassa\Model\Receipt\MarkQuantity OR null</code> | mark_quantity  | Дробное количество маркированного товара |

**Returns:** self - 


<a name="method_setMeasure" class="anchor"></a>
#### public setMeasure() : self

```php
public setMeasure(string|null $measure) : self
```

**Summary**

Устанавливает меру количества предмета расчета.

**Details:**
* Inherited From: [\YooKassa\Model\Receipt\ReceiptItem](../classes/YooKassa-Model-Receipt-ReceiptItem.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | measure  | Мера количества предмета расчета |

**Returns:** self - 


<a name="method_setPaymentMode" class="anchor"></a>
#### public setPaymentMode() : self

```php
public setPaymentMode(string|null $payment_mode = null) : self
```

**Summary**

Устанавливает признак способа расчета.

**Details:**
* Inherited From: [\YooKassa\Model\Receipt\ReceiptItem](../classes/YooKassa-Model-Receipt-ReceiptItem.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | payment_mode  | Признак способа расчета |

**Returns:** self - 


<a name="method_setPaymentSubject" class="anchor"></a>
#### public setPaymentSubject() : self

```php
public setPaymentSubject(null|string $payment_subject = null) : self
```

**Summary**

Устанавливает признак предмета расчета.

**Details:**
* Inherited From: [\YooKassa\Model\Receipt\ReceiptItem](../classes/YooKassa-Model-Receipt-ReceiptItem.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">null OR string</code> | payment_subject  | Признак предмета расчета |

**Returns:** self - 


<a name="method_setPaymentSubjectIndustryDetails" class="anchor"></a>
#### public setPaymentSubjectIndustryDetails() : self

```php
public setPaymentSubjectIndustryDetails(array|\YooKassa\Common\ListObjectInterface|null $payment_subject_industry_details = null) : self
```

**Summary**

Устанавливает отраслевой реквизит чека.

**Details:**
* Inherited From: [\YooKassa\Model\Receipt\ReceiptItem](../classes/YooKassa-Model-Receipt-ReceiptItem.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">array OR \YooKassa\Common\ListObjectInterface OR null</code> | payment_subject_industry_details  | Отраслевой реквизит чека |

**Returns:** self - 


<a name="method_setPrice" class="anchor"></a>
#### public setPrice() : self

```php
public setPrice(\YooKassa\Model\AmountInterface|array $amount) : self
```

**Summary**

Устанавливает цену товара.

**Details:**
* Inherited From: [\YooKassa\Model\Receipt\ReceiptItem](../classes/YooKassa-Model-Receipt-ReceiptItem.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">\YooKassa\Model\AmountInterface OR array</code> | amount  | Цена товара |

**Returns:** self - 


<a name="method_setProductCode" class="anchor"></a>
#### public setProductCode() : self

```php
public setProductCode(\YooKassa\Helpers\ProductCode|string $product_code) : self
```

**Summary**

Устанавливает код товара — уникальный номер, который присваивается экземпляру товара при маркировке.

**Details:**
* Inherited From: [\YooKassa\Model\Receipt\ReceiptItem](../classes/YooKassa-Model-Receipt-ReceiptItem.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">\YooKassa\Helpers\ProductCode OR string</code> | product_code  | Код товара |

**Returns:** self - 


<a name="method_setQuantity" class="anchor"></a>
#### public setQuantity() : self

```php
public setQuantity(float|null $quantity) : self
```

**Summary**

Устанавливает количество покупаемого товара.

**Details:**
* Inherited From: [\YooKassa\Model\Receipt\ReceiptItem](../classes/YooKassa-Model-Receipt-ReceiptItem.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">float OR null</code> | quantity  | Количество |

**Returns:** self - 


<a name="method_setSupplier" class="anchor"></a>
#### public setSupplier() : self

```php
public setSupplier(array|\YooKassa\Model\Receipt\SupplierInterface|null $supplier = null) : self
```

**Summary**

Устанавливает информацию о поставщике товара или услуги.

**Details:**
* Inherited From: [\YooKassa\Model\Receipt\ReceiptItem](../classes/YooKassa-Model-Receipt-ReceiptItem.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">array OR \YooKassa\Model\Receipt\SupplierInterface OR null</code> | supplier  | Информация о поставщике товара или услуги |

**Returns:** self - 


<a name="method_setVatCode" class="anchor"></a>
#### public setVatCode() : self

```php
public setVatCode(?int $vat_code = null) : self
```

**Summary**

Устанавливает ставку НДС

**Details:**
* Inherited From: [\YooKassa\Model\Receipt\ReceiptItem](../classes/YooKassa-Model-Receipt-ReceiptItem.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">?int</code> | vat_code  |  |

**Returns:** self - 


<a name="method_toArray" class="anchor"></a>
#### public toArray() : array

```php
public toArray() : array
```

**Summary**

Возвращает ассоциативный массив со свойствами текущего объекта для его дальнейшей JSON сериализации
Является алиасом метода AbstractObject::jsonSerialize().

**Details:**
* Inherited From: [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)

**Returns:** array - Ассоциативный массив со свойствами текущего объекта


<a name="method_getUnknownProperties" class="anchor"></a>
#### protected getUnknownProperties() : array

```php
protected getUnknownProperties() : array
```

**Summary**

Возвращает массив свойств которые не существуют, но были заданы у объекта.

**Details:**
* Inherited From: [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)

**Returns:** array - Ассоциативный массив с не существующими у текущего объекта свойствами


<a name="method_validatePropertyValue" class="anchor"></a>
#### protected validatePropertyValue() : mixed

```php
protected validatePropertyValue(string $propertyName, mixed $propertyValue) : mixed
```

**Details:**
* Inherited From: [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string</code> | propertyName  |  |
| <code lang="php">mixed</code> | propertyValue  |  |

**Returns:** mixed - 



---

### Top Namespaces

* [\YooKassa](../namespaces/yookassa.md)

---

### Reports
* [Errors - 0](../reports/errors.md)
* [Markers - 0](../reports/markers.md)
* [Deprecated - 40](../reports/deprecated.md)

---

This document was automatically generated from source code comments on 2025-09-04 using [phpDocumentor](http://www.phpdoc.org/)

&copy; 2025 YooMoney