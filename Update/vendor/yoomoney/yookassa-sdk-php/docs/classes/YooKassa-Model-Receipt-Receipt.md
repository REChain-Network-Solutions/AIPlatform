# [YooKassa API SDK](../home.md)

# Class: \YooKassa\Model\Receipt\Receipt
### Namespace: [\YooKassa\Model\Receipt](../namespaces/yookassa-model-receipt.md)
---
**Summary:**

Класс данных для формирования чека в онлайн-кассе (для соблюдения 54-ФЗ).


---
### Constants
* No constants found

---
### Properties
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| public | [$additional_user_props](../classes/YooKassa-Model-Receipt-Receipt.md#property_additional_user_props) |  | Дополнительный реквизит пользователя (тег в 54 ФЗ — 1084) |
| public | [$additionalUserProps](../classes/YooKassa-Model-Receipt-Receipt.md#property_additionalUserProps) |  | Дополнительный реквизит пользователя (тег в 54 ФЗ — 1084) |
| public | [$customer](../classes/YooKassa-Model-Receipt-Receipt.md#property_customer) |  | Информация о плательщике |
| public | [$items](../classes/YooKassa-Model-Receipt-Receipt.md#property_items) |  | Список товаров в заказе. Для чеков по 54-ФЗ: если используете Чеки от ЮKassa, можно передать максимум 80 товаров, если используете стороннюю онлайн-кассу, максимум 100 товаров.  Для чеков самозанятых — максимум 6 товаров. |
| public | [$receipt_industry_details](../classes/YooKassa-Model-Receipt-Receipt.md#property_receipt_industry_details) |  | Отраслевой реквизит чека (тег в 54 ФЗ — 1261) |
| public | [$receipt_operational_details](../classes/YooKassa-Model-Receipt-Receipt.md#property_receipt_operational_details) |  | Операционный реквизит чека (тег в 54 ФЗ — 1270) |
| public | [$receiptIndustryDetails](../classes/YooKassa-Model-Receipt-Receipt.md#property_receiptIndustryDetails) |  | Отраслевой реквизит чека (тег в 54 ФЗ — 1261) |
| public | [$receiptOperationalDetails](../classes/YooKassa-Model-Receipt-Receipt.md#property_receiptOperationalDetails) |  | Операционный реквизит чека (тег в 54 ФЗ — 1270) |
| public | [$settlements](../classes/YooKassa-Model-Receipt-Receipt.md#property_settlements) |  | Массив оплат, обеспечивающих выдачу товара |
| public | [$shipping_items](../classes/YooKassa-Model-Receipt-Receipt.md#property_shipping_items) |  | Список товаров в заказе, являющихся доставкой |
| public | [$shippingItems](../classes/YooKassa-Model-Receipt-Receipt.md#property_shippingItems) |  | Список товаров в заказе, являющихся доставкой |
| public | [$tax_system_code](../classes/YooKassa-Model-Receipt-Receipt.md#property_tax_system_code) |  | Код системы налогообложения. Число 1-6. |
| public | [$taxSystemCode](../classes/YooKassa-Model-Receipt-Receipt.md#property_taxSystemCode) |  | Код системы налогообложения. Число 1-6. |

---
### Methods
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| public | [__construct()](../classes/YooKassa-Common-AbstractObject.md#method___construct) |  | AbstractObject constructor. |
| public | [__get()](../classes/YooKassa-Common-AbstractObject.md#method___get) |  | Возвращает значение свойства. |
| public | [__isset()](../classes/YooKassa-Common-AbstractObject.md#method___isset) |  | Проверяет наличие свойства. |
| public | [__set()](../classes/YooKassa-Common-AbstractObject.md#method___set) |  | Устанавливает значение свойства. |
| public | [__unset()](../classes/YooKassa-Common-AbstractObject.md#method___unset) |  | Удаляет свойство. |
| public | [addItem()](../classes/YooKassa-Model-Receipt-Receipt.md#method_addItem) |  | Добавляет товар в чек. |
| public | [addReceiptIndustryDetails()](../classes/YooKassa-Model-Receipt-Receipt.md#method_addReceiptIndustryDetails) |  | Добавляет отраслевой реквизит чека. |
| public | [addSettlement()](../classes/YooKassa-Model-Receipt-Receipt.md#method_addSettlement) |  | Добавляет оплату в чек. |
| public | [fromArray()](../classes/YooKassa-Common-AbstractObject.md#method_fromArray) |  | Устанавливает значения свойств текущего объекта из массива. |
| public | [getAdditionalUserProps()](../classes/YooKassa-Model-Receipt-Receipt.md#method_getAdditionalUserProps) |  | Возвращает дополнительный реквизит пользователя. |
| public | [getAmountValue()](../classes/YooKassa-Model-Receipt-Receipt.md#method_getAmountValue) |  | Возвращает стоимость заказа исходя из состава чека. |
| public | [getCustomer()](../classes/YooKassa-Model-Receipt-Receipt.md#method_getCustomer) |  | Возвращает информацию о плательщике. |
| public | [getItems()](../classes/YooKassa-Model-Receipt-Receipt.md#method_getItems) |  | Возвращает список позиций в текущем чеке. |
| public | [getObjectId()](../classes/YooKassa-Model-Receipt-Receipt.md#method_getObjectId) |  | Возвращает Id объекта чека. |
| public | [getReceiptIndustryDetails()](../classes/YooKassa-Model-Receipt-Receipt.md#method_getReceiptIndustryDetails) |  | Возвращает отраслевой реквизит чека. |
| public | [getReceiptOperationalDetails()](../classes/YooKassa-Model-Receipt-Receipt.md#method_getReceiptOperationalDetails) |  | Возвращает операционный реквизит чека. |
| public | [getSettlements()](../classes/YooKassa-Model-Receipt-Receipt.md#method_getSettlements) |  | Возвращает массив оплат, обеспечивающих выдачу товара. |
| public | [getShippingAmountValue()](../classes/YooKassa-Model-Receipt-Receipt.md#method_getShippingAmountValue) |  | Возвращает стоимость доставки исходя из состава чека. |
| public | [getShippingItems()](../classes/YooKassa-Model-Receipt-Receipt.md#method_getShippingItems) |  | Возвращает список товаров в заказе, являющихся доставкой |
| public | [getTaxSystemCode()](../classes/YooKassa-Model-Receipt-Receipt.md#method_getTaxSystemCode) |  | Возвращает код системы налогообложения. |
| public | [getValidator()](../classes/YooKassa-Common-AbstractObject.md#method_getValidator) |  |  |
| public | [jsonSerialize()](../classes/YooKassa-Model-Receipt-Receipt.md#method_jsonSerialize) |  | Возвращает ассоциативный массив со свойствами текущего объекта для его дальнейшей JSON сериализации. |
| public | [normalize()](../classes/YooKassa-Model-Receipt-Receipt.md#method_normalize) |  | Подгоняет стоимость товаров в чеке к общей цене заказа. |
| public | [notEmpty()](../classes/YooKassa-Model-Receipt-Receipt.md#method_notEmpty) |  | Проверяет есть ли в чеке хотя бы одна позиция. |
| public | [offsetExists()](../classes/YooKassa-Common-AbstractObject.md#method_offsetExists) |  | Проверяет наличие свойства. |
| public | [offsetGet()](../classes/YooKassa-Common-AbstractObject.md#method_offsetGet) |  | Возвращает значение свойства. |
| public | [offsetSet()](../classes/YooKassa-Common-AbstractObject.md#method_offsetSet) |  | Устанавливает значение свойства. |
| public | [offsetUnset()](../classes/YooKassa-Common-AbstractObject.md#method_offsetUnset) |  | Удаляет свойство. |
| public | [removeItems()](../classes/YooKassa-Model-Receipt-Receipt.md#method_removeItems) |  | Обнуляет список позиций в чеке. |
| public | [setAdditionalUserProps()](../classes/YooKassa-Model-Receipt-Receipt.md#method_setAdditionalUserProps) |  | Устанавливает дополнительный реквизит пользователя. |
| public | [setCustomer()](../classes/YooKassa-Model-Receipt-Receipt.md#method_setCustomer) |  | Устанавливает информацию о плательщике. |
| public | [setItems()](../classes/YooKassa-Model-Receipt-Receipt.md#method_setItems) |  | Устанавливает список позиций в чеке. |
| public | [setReceiptIndustryDetails()](../classes/YooKassa-Model-Receipt-Receipt.md#method_setReceiptIndustryDetails) |  | Устанавливает отраслевой реквизит чека. |
| public | [setReceiptOperationalDetails()](../classes/YooKassa-Model-Receipt-Receipt.md#method_setReceiptOperationalDetails) |  | Устанавливает операционный реквизит чека. |
| public | [setSettlements()](../classes/YooKassa-Model-Receipt-Receipt.md#method_setSettlements) |  | Возвращает массив оплат, обеспечивающих выдачу товара. |
| public | [setTaxSystemCode()](../classes/YooKassa-Model-Receipt-Receipt.md#method_setTaxSystemCode) |  | Устанавливает код системы налогообложения. |
| public | [toArray()](../classes/YooKassa-Common-AbstractObject.md#method_toArray) |  | Возвращает ассоциативный массив со свойствами текущего объекта для его дальнейшей JSON сериализации Является алиасом метода AbstractObject::jsonSerialize(). |
| protected | [getUnknownProperties()](../classes/YooKassa-Common-AbstractObject.md#method_getUnknownProperties) |  | Возвращает массив свойств которые не существуют, но были заданы у объекта. |
| protected | [validatePropertyValue()](../classes/YooKassa-Common-AbstractObject.md#method_validatePropertyValue) |  |  |

---
### Details
* File: [lib/Model/Receipt/Receipt.php](../../lib/Model/Receipt/Receipt.php)
* Package: YooKassa\Model
* Class Hierarchy: 
  * [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)
  * \YooKassa\Model\Receipt\Receipt
* Implements:
  * [\YooKassa\Model\Receipt\ReceiptInterface](../classes/YooKassa-Model-Receipt-ReceiptInterface.md)

* See Also:
  * [](https://yookassa.ru/developers/api)

---
### Tags
| Tag | Version | Description |
| --- | ------- | ----------- |
| category |  | Class |
| author |  | cms@yoomoney.ru |

---
## Properties
<a name="property_additional_user_props"></a>
#### public $additional_user_props : \YooKassa\Model\Receipt\AdditionalUserProps
---
***Description***

Дополнительный реквизит пользователя (тег в 54 ФЗ — 1084)

**Type:** <a href="../classes/YooKassa-Model-Receipt-AdditionalUserProps.html"><abbr title="\YooKassa\Model\Receipt\AdditionalUserProps">AdditionalUserProps</abbr></a>

**Details:**


<a name="property_additionalUserProps"></a>
#### public $additionalUserProps : \YooKassa\Model\Receipt\AdditionalUserProps
---
***Description***

Дополнительный реквизит пользователя (тег в 54 ФЗ — 1084)

**Type:** <a href="../classes/YooKassa-Model-Receipt-AdditionalUserProps.html"><abbr title="\YooKassa\Model\Receipt\AdditionalUserProps">AdditionalUserProps</abbr></a>

**Details:**


<a name="property_customer"></a>
#### public $customer : \YooKassa\Model\Receipt\ReceiptCustomer
---
***Description***

Информация о плательщике

**Type:** <a href="../classes/YooKassa-Model-Receipt-ReceiptCustomer.html"><abbr title="\YooKassa\Model\Receipt\ReceiptCustomer">ReceiptCustomer</abbr></a>

**Details:**


<a name="property_items"></a>
#### public $items : \YooKassa\Common\ListObjectInterface|\YooKassa\Model\Receipt\ReceiptItemInterface[]
---
***Description***

Список товаров в заказе. Для чеков по 54-ФЗ: если используете Чеки от ЮKassa, можно передать максимум 80 товаров, если используете стороннюю онлайн-кассу, максимум 100 товаров.  Для чеков самозанятых — максимум 6 товаров.

**Type:** <a href="../\YooKassa\Common\ListObjectInterface|\YooKassa\Model\Receipt\ReceiptItemInterface[]"><abbr title="\YooKassa\Common\ListObjectInterface|\YooKassa\Model\Receipt\ReceiptItemInterface[]">ReceiptItemInterface[]</abbr></a>

**Details:**


<a name="property_receipt_industry_details"></a>
#### public $receipt_industry_details : \YooKassa\Common\ListObjectInterface|\YooKassa\Model\Receipt\IndustryDetails[]
---
***Description***

Отраслевой реквизит чека (тег в 54 ФЗ — 1261)

**Type:** <a href="../\YooKassa\Common\ListObjectInterface|\YooKassa\Model\Receipt\IndustryDetails[]"><abbr title="\YooKassa\Common\ListObjectInterface|\YooKassa\Model\Receipt\IndustryDetails[]">IndustryDetails[]</abbr></a>

**Details:**


<a name="property_receipt_operational_details"></a>
#### public $receipt_operational_details : \YooKassa\Model\Receipt\OperationalDetails
---
***Description***

Операционный реквизит чека (тег в 54 ФЗ — 1270)

**Type:** <a href="../classes/YooKassa-Model-Receipt-OperationalDetails.html"><abbr title="\YooKassa\Model\Receipt\OperationalDetails">OperationalDetails</abbr></a>

**Details:**


<a name="property_receiptIndustryDetails"></a>
#### public $receiptIndustryDetails : \YooKassa\Common\ListObjectInterface|\YooKassa\Model\Receipt\IndustryDetails[]
---
***Description***

Отраслевой реквизит чека (тег в 54 ФЗ — 1261)

**Type:** <a href="../\YooKassa\Common\ListObjectInterface|\YooKassa\Model\Receipt\IndustryDetails[]"><abbr title="\YooKassa\Common\ListObjectInterface|\YooKassa\Model\Receipt\IndustryDetails[]">IndustryDetails[]</abbr></a>

**Details:**


<a name="property_receiptOperationalDetails"></a>
#### public $receiptOperationalDetails : \YooKassa\Model\Receipt\OperationalDetails
---
***Description***

Операционный реквизит чека (тег в 54 ФЗ — 1270)

**Type:** <a href="../classes/YooKassa-Model-Receipt-OperationalDetails.html"><abbr title="\YooKassa\Model\Receipt\OperationalDetails">OperationalDetails</abbr></a>

**Details:**


<a name="property_settlements"></a>
#### public $settlements : \YooKassa\Common\ListObjectInterface|\YooKassa\Model\Receipt\SettlementInterface[]
---
***Description***

Массив оплат, обеспечивающих выдачу товара

**Type:** <a href="../\YooKassa\Common\ListObjectInterface|\YooKassa\Model\Receipt\SettlementInterface[]"><abbr title="\YooKassa\Common\ListObjectInterface|\YooKassa\Model\Receipt\SettlementInterface[]">SettlementInterface[]</abbr></a>

**Details:**


<a name="property_shipping_items"></a>
#### public $shipping_items : \YooKassa\Common\ListObjectInterface|\YooKassa\Model\Receipt\ReceiptItemInterface[]
---
***Description***

Список товаров в заказе, являющихся доставкой

**Type:** <a href="../\YooKassa\Common\ListObjectInterface|\YooKassa\Model\Receipt\ReceiptItemInterface[]"><abbr title="\YooKassa\Common\ListObjectInterface|\YooKassa\Model\Receipt\ReceiptItemInterface[]">ReceiptItemInterface[]</abbr></a>

**Details:**


<a name="property_shippingItems"></a>
#### public $shippingItems : \YooKassa\Common\ListObjectInterface|\YooKassa\Model\Receipt\ReceiptItemInterface[]
---
***Description***

Список товаров в заказе, являющихся доставкой

**Type:** <a href="../\YooKassa\Common\ListObjectInterface|\YooKassa\Model\Receipt\ReceiptItemInterface[]"><abbr title="\YooKassa\Common\ListObjectInterface|\YooKassa\Model\Receipt\ReceiptItemInterface[]">ReceiptItemInterface[]</abbr></a>

**Details:**


<a name="property_tax_system_code"></a>
#### public $tax_system_code : int
---
***Description***

Код системы налогообложения. Число 1-6.

**Type:** <a href="../int"><abbr title="int">int</abbr></a>

**Details:**


<a name="property_taxSystemCode"></a>
#### public $taxSystemCode : int
---
***Description***

Код системы налогообложения. Число 1-6.

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


<a name="method_addItem" class="anchor"></a>
#### public addItem() : void

```php
public addItem(\YooKassa\Model\Receipt\ReceiptItemInterface[]|\YooKassa\Common\ListObjectInterface $value) : void
```

**Summary**

Добавляет товар в чек.

**Details:**
* Inherited From: [\YooKassa\Model\Receipt\Receipt](../classes/YooKassa-Model-Receipt-Receipt.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">\YooKassa\Model\Receipt\ReceiptItemInterface[] OR \YooKassa\Common\ListObjectInterface</code> | value  | Объект добавляемой в чек позиции |

**Returns:** void - 


<a name="method_addReceiptIndustryDetails" class="anchor"></a>
#### public addReceiptIndustryDetails() : self

```php
public addReceiptIndustryDetails(\YooKassa\Model\Receipt\IndustryDetails|array $value) : self
```

**Summary**

Добавляет отраслевой реквизит чека.

**Details:**
* Inherited From: [\YooKassa\Model\Receipt\Receipt](../classes/YooKassa-Model-Receipt-Receipt.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">\YooKassa\Model\Receipt\IndustryDetails OR array</code> | value  | Отраслевой реквизит чека. |

**Returns:** self - 


<a name="method_addSettlement" class="anchor"></a>
#### public addSettlement() : self

```php
public addSettlement(\YooKassa\Model\Receipt\SettlementInterface $value) : self
```

**Summary**

Добавляет оплату в чек.

**Details:**
* Inherited From: [\YooKassa\Model\Receipt\Receipt](../classes/YooKassa-Model-Receipt-Receipt.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">\YooKassa\Model\Receipt\SettlementInterface</code> | value  | Оплата |

**Returns:** self - 


<a name="method_fromArray" class="anchor"></a>
#### public fromArray() : void

```php
public fromArray(array|\Traversable $sourceArray) : void
```

**Summary**

Устанавливает значения свойств текущего объекта из массива.

**Details:**
* Inherited From: [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">array OR \Traversable</code> | sourceArray  | Ассоциативный массив с настройками |

**Returns:** void - 


<a name="method_getAdditionalUserProps" class="anchor"></a>
#### public getAdditionalUserProps() : \YooKassa\Model\Receipt\AdditionalUserProps|null

```php
public getAdditionalUserProps() : \YooKassa\Model\Receipt\AdditionalUserProps|null
```

**Summary**

Возвращает дополнительный реквизит пользователя.

**Details:**
* Inherited From: [\YooKassa\Model\Receipt\Receipt](../classes/YooKassa-Model-Receipt-Receipt.md)

**Returns:** \YooKassa\Model\Receipt\AdditionalUserProps|null - Дополнительный реквизит пользователя


<a name="method_getAmountValue" class="anchor"></a>
#### public getAmountValue() : int

```php
public getAmountValue(bool $withShipping = true) : int
```

**Summary**

Возвращает стоимость заказа исходя из состава чека.

**Details:**
* Inherited From: [\YooKassa\Model\Receipt\Receipt](../classes/YooKassa-Model-Receipt-Receipt.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">bool</code> | withShipping  | Добавить ли к стоимости заказа стоимость доставки |

**Returns:** int - Общая стоимость заказа в центах/копейках


<a name="method_getCustomer" class="anchor"></a>
#### public getCustomer() : \YooKassa\Model\Receipt\ReceiptCustomer

```php
public getCustomer() : \YooKassa\Model\Receipt\ReceiptCustomer
```

**Summary**

Возвращает информацию о плательщике.

**Details:**
* Inherited From: [\YooKassa\Model\Receipt\Receipt](../classes/YooKassa-Model-Receipt-Receipt.md)

**Returns:** \YooKassa\Model\Receipt\ReceiptCustomer - Информация о плательщике


<a name="method_getItems" class="anchor"></a>
#### public getItems() : \YooKassa\Model\Receipt\ReceiptItemInterface[]|\YooKassa\Common\ListObjectInterface

```php
public getItems() : \YooKassa\Model\Receipt\ReceiptItemInterface[]|\YooKassa\Common\ListObjectInterface
```

**Summary**

Возвращает список позиций в текущем чеке.

**Details:**
* Inherited From: [\YooKassa\Model\Receipt\Receipt](../classes/YooKassa-Model-Receipt-Receipt.md)

**Returns:** \YooKassa\Model\Receipt\ReceiptItemInterface[]|\YooKassa\Common\ListObjectInterface - Список товаров в заказе


<a name="method_getObjectId" class="anchor"></a>
#### public getObjectId() : null|string

```php
public getObjectId() : null|string
```

**Summary**

Возвращает Id объекта чека.

**Details:**
* Inherited From: [\YooKassa\Model\Receipt\Receipt](../classes/YooKassa-Model-Receipt-Receipt.md)

**Returns:** null|string - Id объекта чека


<a name="method_getReceiptIndustryDetails" class="anchor"></a>
#### public getReceiptIndustryDetails() : \YooKassa\Model\Receipt\IndustryDetails[]|\YooKassa\Common\ListObjectInterface

```php
public getReceiptIndustryDetails() : \YooKassa\Model\Receipt\IndustryDetails[]|\YooKassa\Common\ListObjectInterface
```

**Summary**

Возвращает отраслевой реквизит чека.

**Details:**
* Inherited From: [\YooKassa\Model\Receipt\Receipt](../classes/YooKassa-Model-Receipt-Receipt.md)

**Returns:** \YooKassa\Model\Receipt\IndustryDetails[]|\YooKassa\Common\ListObjectInterface - Отраслевой реквизит чека


<a name="method_getReceiptOperationalDetails" class="anchor"></a>
#### public getReceiptOperationalDetails() : \YooKassa\Model\Receipt\OperationalDetails|null

```php
public getReceiptOperationalDetails() : \YooKassa\Model\Receipt\OperationalDetails|null
```

**Summary**

Возвращает операционный реквизит чека.

**Details:**
* Inherited From: [\YooKassa\Model\Receipt\Receipt](../classes/YooKassa-Model-Receipt-Receipt.md)

**Returns:** \YooKassa\Model\Receipt\OperationalDetails|null - Операционный реквизит чека


<a name="method_getSettlements" class="anchor"></a>
#### public getSettlements() : \YooKassa\Model\Receipt\SettlementInterface[]|\YooKassa\Common\ListObjectInterface

```php
public getSettlements() : \YooKassa\Model\Receipt\SettlementInterface[]|\YooKassa\Common\ListObjectInterface
```

**Summary**

Возвращает массив оплат, обеспечивающих выдачу товара.

**Details:**
* Inherited From: [\YooKassa\Model\Receipt\Receipt](../classes/YooKassa-Model-Receipt-Receipt.md)

**Returns:** \YooKassa\Model\Receipt\SettlementInterface[]|\YooKassa\Common\ListObjectInterface - Массив оплат, обеспечивающих выдачу товара


<a name="method_getShippingAmountValue" class="anchor"></a>
#### public getShippingAmountValue() : int

```php
public getShippingAmountValue() : int
```

**Summary**

Возвращает стоимость доставки исходя из состава чека.

**Details:**
* Inherited From: [\YooKassa\Model\Receipt\Receipt](../classes/YooKassa-Model-Receipt-Receipt.md)

**Returns:** int - Стоимость доставки из состава чека в центах/копейках


<a name="method_getShippingItems" class="anchor"></a>
#### public getShippingItems() : \YooKassa\Model\Receipt\ReceiptItemInterface[]|\YooKassa\Common\ListObjectInterface

```php
public getShippingItems() : \YooKassa\Model\Receipt\ReceiptItemInterface[]|\YooKassa\Common\ListObjectInterface
```

**Summary**

Возвращает список товаров в заказе, являющихся доставкой

**Details:**
* Inherited From: [\YooKassa\Model\Receipt\Receipt](../classes/YooKassa-Model-Receipt-Receipt.md)

**Returns:** \YooKassa\Model\Receipt\ReceiptItemInterface[]|\YooKassa\Common\ListObjectInterface - Список товаров в заказе, являющихся доставкой


<a name="method_getTaxSystemCode" class="anchor"></a>
#### public getTaxSystemCode() : int|null

```php
public getTaxSystemCode() : int|null
```

**Summary**

Возвращает код системы налогообложения.

**Details:**
* Inherited From: [\YooKassa\Model\Receipt\Receipt](../classes/YooKassa-Model-Receipt-Receipt.md)

**Returns:** int|null - Код системы налогообложения. Число 1-6.


<a name="method_getValidator" class="anchor"></a>
#### public getValidator() : \YooKassa\Validator\Validator

```php
public getValidator() : \YooKassa\Validator\Validator
```

**Details:**
* Inherited From: [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)

**Returns:** \YooKassa\Validator\Validator - 


<a name="method_jsonSerialize" class="anchor"></a>
#### public jsonSerialize() : array

```php
public jsonSerialize() : array
```

**Summary**

Возвращает ассоциативный массив со свойствами текущего объекта для его дальнейшей JSON сериализации.

**Details:**
* Inherited From: [\YooKassa\Model\Receipt\Receipt](../classes/YooKassa-Model-Receipt-Receipt.md)

**Returns:** array - Ассоциативный массив со свойствами текущего объекта


<a name="method_normalize" class="anchor"></a>
#### public normalize() : void

```php
public normalize(\YooKassa\Model\AmountInterface $orderAmount, bool $withShipping = false) : void
```

**Summary**

Подгоняет стоимость товаров в чеке к общей цене заказа.

**Details:**
* Inherited From: [\YooKassa\Model\Receipt\Receipt](../classes/YooKassa-Model-Receipt-Receipt.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">\YooKassa\Model\AmountInterface</code> | orderAmount  | Общая стоимость заказа |
| <code lang="php">bool</code> | withShipping  | Поменять ли заодно и цену доставки |

**Returns:** void - 


<a name="method_notEmpty" class="anchor"></a>
#### public notEmpty() : bool

```php
public notEmpty() : bool
```

**Summary**

Проверяет есть ли в чеке хотя бы одна позиция.

**Details:**
* Inherited From: [\YooKassa\Model\Receipt\Receipt](../classes/YooKassa-Model-Receipt-Receipt.md)

**Returns:** bool - True если чек не пуст, false если в чеке нет ни одной позиции


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


<a name="method_removeItems" class="anchor"></a>
#### public removeItems() : self

```php
public removeItems() : self
```

**Summary**

Обнуляет список позиций в чеке.

**Description**

Если до этого в чеке уже были установлены значения, они удаляются.

**Details:**
* Inherited From: [\YooKassa\Model\Receipt\Receipt](../classes/YooKassa-Model-Receipt-Receipt.md)

**Returns:** self - 


<a name="method_setAdditionalUserProps" class="anchor"></a>
#### public setAdditionalUserProps() : self

```php
public setAdditionalUserProps(\YooKassa\Model\Receipt\AdditionalUserProps|array $additional_user_props = null) : self
```

**Summary**

Устанавливает дополнительный реквизит пользователя.

**Details:**
* Inherited From: [\YooKassa\Model\Receipt\Receipt](../classes/YooKassa-Model-Receipt-Receipt.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">\YooKassa\Model\Receipt\AdditionalUserProps OR array</code> | additional_user_props  | Дополнительный реквизит пользователя |

**Returns:** self - 


<a name="method_setCustomer" class="anchor"></a>
#### public setCustomer() : self

```php
public setCustomer(\YooKassa\Model\Receipt\ReceiptCustomer|array|null $customer = null) : self
```

**Summary**

Устанавливает информацию о плательщике.

**Details:**
* Inherited From: [\YooKassa\Model\Receipt\Receipt](../classes/YooKassa-Model-Receipt-Receipt.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">\YooKassa\Model\Receipt\ReceiptCustomer OR array OR null</code> | customer  |  |

**Returns:** self - 


<a name="method_setItems" class="anchor"></a>
#### public setItems() : self

```php
public setItems(array|\YooKassa\Common\ListObjectInterface $items) : self
```

**Summary**

Устанавливает список позиций в чеке.

**Description**

Если до этого в чеке уже были установлены значения, они удаляются и полностью заменяются переданным списком
позиций. Все передаваемые значения в массиве позиций должны быть объектами класса, реализующего интерфейс
ReceiptItemInterface, в противном случае будет выброшено исключение InvalidPropertyValueTypeException.

**Details:**
* Inherited From: [\YooKassa\Model\Receipt\Receipt](../classes/YooKassa-Model-Receipt-Receipt.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">array OR \YooKassa\Common\ListObjectInterface</code> | items  | Список товаров в заказе: для [Чеков от ЮKassa](https://yookassa.ru/developers/payment-acceptance/receipts/54fz/yoomoney/basics) — не более 80 товаров, для [сторонних онлайн-касс](https://yookassa.ru/developers/payment-acceptance/receipts/54fz/other-services/basics) — не более 100 товаров |

##### Throws:
| Type | Description |
| ---- | ----------- |
| \YooKassa\Validator\Exceptions\EmptyPropertyValueException | Выбрасывается если передали пустой массив значений |
| \YooKassa\Validator\Exceptions\InvalidPropertyValueTypeException | Выбрасывается если в качестве значения был передан не массив и не итератор, либо если одно из переданных значений не реализует интерфейс ReceiptItemInterface |

**Returns:** self - 


<a name="method_setReceiptIndustryDetails" class="anchor"></a>
#### public setReceiptIndustryDetails() : self

```php
public setReceiptIndustryDetails(array|\YooKassa\Common\ListObjectInterface|null $receipt_industry_details = null) : self
```

**Summary**

Устанавливает отраслевой реквизит чека.

**Details:**
* Inherited From: [\YooKassa\Model\Receipt\Receipt](../classes/YooKassa-Model-Receipt-Receipt.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">array OR \YooKassa\Common\ListObjectInterface OR null</code> | receipt_industry_details  | Отраслевой реквизит чека |

##### Throws:
| Type | Description |
| ---- | ----------- |
| \YooKassa\Validator\Exceptions\InvalidPropertyValueTypeException | Выбрасывается если переданный аргумент - не массив |

**Returns:** self - 


<a name="method_setReceiptOperationalDetails" class="anchor"></a>
#### public setReceiptOperationalDetails() : self

```php
public setReceiptOperationalDetails(array|\YooKassa\Model\Receipt\OperationalDetails|null $receipt_operational_details = null) : self
```

**Summary**

Устанавливает операционный реквизит чека.

**Details:**
* Inherited From: [\YooKassa\Model\Receipt\Receipt](../classes/YooKassa-Model-Receipt-Receipt.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">array OR \YooKassa\Model\Receipt\OperationalDetails OR null</code> | receipt_operational_details  | Операционный реквизит чека |

##### Throws:
| Type | Description |
| ---- | ----------- |
| \YooKassa\Validator\Exceptions\InvalidPropertyValueTypeException | Выбрасывается если переданный аргумент - не массив |

**Returns:** self - 


<a name="method_setSettlements" class="anchor"></a>
#### public setSettlements() : self

```php
public setSettlements(\YooKassa\Common\ListObjectInterface|array|null $settlements = null) : self
```

**Summary**

Возвращает массив оплат, обеспечивающих выдачу товара.

**Details:**
* Inherited From: [\YooKassa\Model\Receipt\Receipt](../classes/YooKassa-Model-Receipt-Receipt.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">\YooKassa\Common\ListObjectInterface OR array OR null</code> | settlements  |  |

**Returns:** self - 


<a name="method_setTaxSystemCode" class="anchor"></a>
#### public setTaxSystemCode() : self

```php
public setTaxSystemCode(int|null $tax_system_code) : self
```

**Summary**

Устанавливает код системы налогообложения.

**Details:**
* Inherited From: [\YooKassa\Model\Receipt\Receipt](../classes/YooKassa-Model-Receipt-Receipt.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">int OR null</code> | tax_system_code  | Код системы налогообложения. Число 1-6 |

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