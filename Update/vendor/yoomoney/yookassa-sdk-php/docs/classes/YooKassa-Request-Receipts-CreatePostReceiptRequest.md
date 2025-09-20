# [YooKassa API SDK](../home.md)

# Class: \YooKassa\Request\Receipts\CreatePostReceiptRequest
### Namespace: [\YooKassa\Request\Receipts](../namespaces/yookassa-request-receipts.md)
---
**Summary:**

Класс объекта запроса к API на создание чека.


---
### Examples
Пример использования билдера

```php
try {
    $receiptBuilder = \YooKassa\Request\Receipts\CreatePostReceiptRequest::builder();
    $receiptBuilder->setType(\YooKassa\Model\Receipt\ReceiptType::PAYMENT)
        ->setObjectId('24b94598-000f-5000-9000-1b68e7b15f3f', \YooKassa\Model\Receipt\ReceiptType::PAYMENT) // payment_id
        ->setCustomer([
            'email' => 'john.doe@merchant.com',
            'phone' => '71111111111',
        ])
        ->setItems([
            [
                'description' => 'Платок Gucci',
                'quantity' => '1.00',
                'amount' => [
                    'value' => '3000.00',
                    'currency' => 'RUB',
                ],
                'vat_code' => 2,
                'payment_mode' => 'full_payment',
                'payment_subject' => 'commodity',
            ],
        ])
        ->addSettlement([
            [
                'type' => 'prepayment',
                'amount' => [
                    'value' => 100.00,
                    'currency' => 'RUB',
                ],
            ],
        ])
        ->setSend(true)
    ;

    // Создаем объект запроса
    $request = $receiptBuilder->build();

    // Можно изменить данные, если нужно
    $request->setOnBehalfOf('159753');
    $request->getitems()->add(new \YooKassa\Model\Receipt\ReceiptItem([
        'description' => 'Платок Gucci Новый',
        'quantity' => '1.00',
        'amount' => [
            'value' => '3500.00',
            'currency' => 'RUB',
        ],
        'vat_code' => 2,
        'payment_mode' => 'full_payment',
        'payment_subject' => 'commodity',
    ]));

    $idempotenceKey = uniqid('', true);
    $response = $client->createReceipt($request, $idempotenceKey);
} catch (Exception $e) {
    $response = $e;
}

var_dump($response);

```

---
### Constants
* No constants found

---
### Properties
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| public | [$additional_user_props](../classes/YooKassa-Request-Receipts-CreatePostReceiptRequest.md#property_additional_user_props) |  | Дополнительный реквизит пользователя. |
| public | [$additionalUserProps](../classes/YooKassa-Request-Receipts-CreatePostReceiptRequest.md#property_additionalUserProps) |  | Дополнительный реквизит пользователя. |
| public | [$customer](../classes/YooKassa-Request-Receipts-CreatePostReceiptRequest.md#property_customer) |  | Информация о плательщике |
| public | [$items](../classes/YooKassa-Request-Receipts-CreatePostReceiptRequest.md#property_items) |  | Список товаров в заказе: для [Чеков от ЮKassa](https://yookassa.ru/developers/payment-acceptance/receipts/54fz/yoomoney/basics) — не более 80 товаров, для [сторонних онлайн-касс](https://yookassa.ru/developers/payment-acceptance/receipts/54fz/other-services/basics) — не более 100 товаров. |
| public | [$object_id](../classes/YooKassa-Request-Receipts-CreatePostReceiptRequest.md#property_object_id) |  | Идентификатор объекта оплаты |
| public | [$object_type](../classes/YooKassa-Request-Receipts-CreatePostReceiptRequest.md#property_object_type) |  | Тип объекта: приход "payment" или возврат "refund". |
| public | [$objectId](../classes/YooKassa-Request-Receipts-CreatePostReceiptRequest.md#property_objectId) |  | Идентификатор объекта оплаты |
| public | [$objectType](../classes/YooKassa-Request-Receipts-CreatePostReceiptRequest.md#property_objectType) |  | Тип объекта: приход "payment" или возврат "refund". |
| public | [$on_behalf_of](../classes/YooKassa-Request-Receipts-CreatePostReceiptRequest.md#property_on_behalf_of) |  | Идентификатор магазина в ЮKassa. |
| public | [$onBehalfOf](../classes/YooKassa-Request-Receipts-CreatePostReceiptRequest.md#property_onBehalfOf) |  | Идентификатор магазина в ЮKassa. |
| public | [$receipt_industry_details](../classes/YooKassa-Request-Receipts-CreatePostReceiptRequest.md#property_receipt_industry_details) |  | Отраслевой реквизит предмета расчета. |
| public | [$receipt_operational_details](../classes/YooKassa-Request-Receipts-CreatePostReceiptRequest.md#property_receipt_operational_details) |  | Операционный реквизит чека. |
| public | [$receiptIndustryDetails](../classes/YooKassa-Request-Receipts-CreatePostReceiptRequest.md#property_receiptIndustryDetails) |  | Отраслевой реквизит предмета расчета. |
| public | [$receiptOperationalDetails](../classes/YooKassa-Request-Receipts-CreatePostReceiptRequest.md#property_receiptOperationalDetails) |  | Операционный реквизит чека. |
| public | [$send](../classes/YooKassa-Request-Receipts-CreatePostReceiptRequest.md#property_send) |  | Формирование чека в онлайн-кассе сразу после создания объекта чека. Сейчас можно передать только значение ~`true`. |
| public | [$settlements](../classes/YooKassa-Request-Receipts-CreatePostReceiptRequest.md#property_settlements) |  | Список платежей |
| public | [$tax_system_code](../classes/YooKassa-Request-Receipts-CreatePostReceiptRequest.md#property_tax_system_code) |  | Система налогообложения магазина (тег в 54 ФЗ — 1055). |
| public | [$taxSystemCode](../classes/YooKassa-Request-Receipts-CreatePostReceiptRequest.md#property_taxSystemCode) |  | Система налогообложения магазина (тег в 54 ФЗ — 1055). |
| public | [$type](../classes/YooKassa-Request-Receipts-CreatePostReceiptRequest.md#property_type) |  | Тип чека в онлайн-кассе: приход "payment" или возврат "refund". |

---
### Methods
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| public | [__construct()](../classes/YooKassa-Common-AbstractObject.md#method___construct) |  | AbstractObject constructor. |
| public | [__get()](../classes/YooKassa-Common-AbstractObject.md#method___get) |  | Возвращает значение свойства. |
| public | [__isset()](../classes/YooKassa-Common-AbstractObject.md#method___isset) |  | Проверяет наличие свойства. |
| public | [__set()](../classes/YooKassa-Common-AbstractObject.md#method___set) |  | Устанавливает значение свойства. |
| public | [__unset()](../classes/YooKassa-Common-AbstractObject.md#method___unset) |  | Удаляет свойство. |
| public | [builder()](../classes/YooKassa-Request-Receipts-CreatePostReceiptRequest.md#method_builder) |  | Возвращает билдер объектов запросов создания платежа. |
| public | [clearValidationError()](../classes/YooKassa-Common-AbstractRequest.md#method_clearValidationError) |  | Очищает статус валидации текущего запроса. |
| public | [fromArray()](../classes/YooKassa-Common-AbstractObject.md#method_fromArray) |  | Устанавливает значения свойств текущего объекта из массива. |
| public | [getAdditionalUserProps()](../classes/YooKassa-Request-Receipts-CreatePostReceiptRequest.md#method_getAdditionalUserProps) |  | Возвращает дополнительный реквизит пользователя. |
| public | [getCustomer()](../classes/YooKassa-Request-Receipts-CreatePostReceiptRequest.md#method_getCustomer) |  | Возвращает информацию о плательщике. |
| public | [getItems()](../classes/YooKassa-Request-Receipts-CreatePostReceiptRequest.md#method_getItems) |  | Возвращает список позиций в текущем чеке. |
| public | [getLastValidationError()](../classes/YooKassa-Common-AbstractRequest.md#method_getLastValidationError) |  | Возвращает последнюю ошибку валидации. |
| public | [getObjectId()](../classes/YooKassa-Request-Receipts-CreatePostReceiptRequest.md#method_getObjectId) |  | Возвращает Id объекта чека. |
| public | [getObjectType()](../classes/YooKassa-Request-Receipts-CreatePostReceiptRequest.md#method_getObjectType) |  | Возвращает тип объекта чека. |
| public | [getOnBehalfOf()](../classes/YooKassa-Request-Receipts-CreatePostReceiptRequest.md#method_getOnBehalfOf) |  | Возвращает идентификатор магазина, от имени которого нужно отправить чек. |
| public | [getReceiptIndustryDetails()](../classes/YooKassa-Request-Receipts-CreatePostReceiptRequest.md#method_getReceiptIndustryDetails) |  | Возвращает отраслевой реквизит чека. |
| public | [getReceiptOperationalDetails()](../classes/YooKassa-Request-Receipts-CreatePostReceiptRequest.md#method_getReceiptOperationalDetails) |  | Возвращает операционный реквизит чека. |
| public | [getSend()](../classes/YooKassa-Request-Receipts-CreatePostReceiptRequest.md#method_getSend) |  | Возвращает признак отложенной отправки чека. |
| public | [getSettlements()](../classes/YooKassa-Request-Receipts-CreatePostReceiptRequest.md#method_getSettlements) |  | Возвращает массив оплат, обеспечивающих выдачу товара. |
| public | [getTaxSystemCode()](../classes/YooKassa-Request-Receipts-CreatePostReceiptRequest.md#method_getTaxSystemCode) |  | Возвращает код системы налогообложения. |
| public | [getType()](../classes/YooKassa-Request-Receipts-CreatePostReceiptRequest.md#method_getType) |  | Возвращает тип чека в онлайн-кассе. |
| public | [getValidator()](../classes/YooKassa-Common-AbstractObject.md#method_getValidator) |  |  |
| public | [hasCustomer()](../classes/YooKassa-Request-Receipts-CreatePostReceiptRequest.md#method_hasCustomer) |  | Проверяет наличие данных о плательщике. |
| public | [jsonSerialize()](../classes/YooKassa-Common-AbstractObject.md#method_jsonSerialize) |  | Возвращает ассоциативный массив со свойствами текущего объекта для его дальнейшей JSON сериализации. |
| public | [notEmpty()](../classes/YooKassa-Request-Receipts-CreatePostReceiptRequest.md#method_notEmpty) |  | Проверяет есть ли в чеке хотя бы одна позиция товаров и оплат |
| public | [offsetExists()](../classes/YooKassa-Common-AbstractObject.md#method_offsetExists) |  | Проверяет наличие свойства. |
| public | [offsetGet()](../classes/YooKassa-Common-AbstractObject.md#method_offsetGet) |  | Возвращает значение свойства. |
| public | [offsetSet()](../classes/YooKassa-Common-AbstractObject.md#method_offsetSet) |  | Устанавливает значение свойства. |
| public | [offsetUnset()](../classes/YooKassa-Common-AbstractObject.md#method_offsetUnset) |  | Удаляет свойство. |
| public | [setAdditionalUserProps()](../classes/YooKassa-Request-Receipts-CreatePostReceiptRequest.md#method_setAdditionalUserProps) |  | Устанавливает дополнительный реквизит пользователя. |
| public | [setCustomer()](../classes/YooKassa-Request-Receipts-CreatePostReceiptRequest.md#method_setCustomer) |  | Устанавливает информацию о плательщике. |
| public | [setItems()](../classes/YooKassa-Request-Receipts-CreatePostReceiptRequest.md#method_setItems) |  | Устанавливает список позиций в чеке. |
| public | [setObjectId()](../classes/YooKassa-Request-Receipts-CreatePostReceiptRequest.md#method_setObjectId) |  | Устанавливает Id объекта чека. |
| public | [setObjectType()](../classes/YooKassa-Request-Receipts-CreatePostReceiptRequest.md#method_setObjectType) |  | Устанавливает тип объекта чека. |
| public | [setOnBehalfOf()](../classes/YooKassa-Request-Receipts-CreatePostReceiptRequest.md#method_setOnBehalfOf) |  | Устанавливает идентификатор магазина, от имени которого нужно отправить чек. |
| public | [setReceiptIndustryDetails()](../classes/YooKassa-Request-Receipts-CreatePostReceiptRequest.md#method_setReceiptIndustryDetails) |  | Устанавливает отраслевой реквизит чека. |
| public | [setReceiptOperationalDetails()](../classes/YooKassa-Request-Receipts-CreatePostReceiptRequest.md#method_setReceiptOperationalDetails) |  | Устанавливает операционный реквизит чека. |
| public | [setSend()](../classes/YooKassa-Request-Receipts-CreatePostReceiptRequest.md#method_setSend) |  | Устанавливает признак отложенной отправки чека. |
| public | [setSettlements()](../classes/YooKassa-Request-Receipts-CreatePostReceiptRequest.md#method_setSettlements) |  | Устанавливает массив оплат, обеспечивающих выдачу товара. |
| public | [setTaxSystemCode()](../classes/YooKassa-Request-Receipts-CreatePostReceiptRequest.md#method_setTaxSystemCode) |  | Устанавливает код системы налогообложения. |
| public | [setType()](../classes/YooKassa-Request-Receipts-CreatePostReceiptRequest.md#method_setType) |  | Устанавливает тип чека в онлайн-кассе. |
| public | [toArray()](../classes/YooKassa-Common-AbstractObject.md#method_toArray) |  | Возвращает ассоциативный массив со свойствами текущего объекта для его дальнейшей JSON сериализации Является алиасом метода AbstractObject::jsonSerialize(). |
| public | [validate()](../classes/YooKassa-Request-Receipts-CreatePostReceiptRequest.md#method_validate) |  | Валидирует текущий запрос, проверяет все ли нужные свойства установлены. |
| protected | [getUnknownProperties()](../classes/YooKassa-Common-AbstractObject.md#method_getUnknownProperties) |  | Возвращает массив свойств которые не существуют, но были заданы у объекта. |
| protected | [setValidationError()](../classes/YooKassa-Common-AbstractRequest.md#method_setValidationError) |  | Устанавливает ошибку валидации. |
| protected | [validatePropertyValue()](../classes/YooKassa-Common-AbstractObject.md#method_validatePropertyValue) |  |  |

---
### Details
* File: [lib/Request/Receipts/CreatePostReceiptRequest.php](../../lib/Request/Receipts/CreatePostReceiptRequest.php)
* Package: YooKassa\Request
* Class Hierarchy:  
  * [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)
  * [\YooKassa\Common\AbstractRequest](../classes/YooKassa-Common-AbstractRequest.md)
  * \YooKassa\Request\Receipts\CreatePostReceiptRequest
* Implements:
  * [\YooKassa\Request\Receipts\CreatePostReceiptRequestInterface](../classes/YooKassa-Request-Receipts-CreatePostReceiptRequestInterface.md)

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

Дополнительный реквизит пользователя.

**Type:** <a href="../classes/YooKassa-Model-Receipt-AdditionalUserProps.html"><abbr title="\YooKassa\Model\Receipt\AdditionalUserProps">AdditionalUserProps</abbr></a>

**Details:**


<a name="property_additionalUserProps"></a>
#### public $additionalUserProps : \YooKassa\Model\Receipt\AdditionalUserProps
---
***Description***

Дополнительный реквизит пользователя.

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
#### public $items : array
---
***Description***

Список товаров в заказе: для [Чеков от ЮKassa](https://yookassa.ru/developers/payment-acceptance/receipts/54fz/yoomoney/basics) — не более 80 товаров, для [сторонних онлайн-касс](https://yookassa.ru/developers/payment-acceptance/receipts/54fz/other-services/basics) — не более 100 товаров.

**Type:** <a href="../array"><abbr title="array">array</abbr></a>

**Details:**


<a name="property_object_id"></a>
#### public $object_id : string
---
***Description***

Идентификатор объекта оплаты

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**


<a name="property_object_type"></a>
#### public $object_type : string
---
***Description***

Тип объекта: приход "payment" или возврат "refund".

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**


<a name="property_objectId"></a>
#### public $objectId : string
---
***Description***

Идентификатор объекта оплаты

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**


<a name="property_objectType"></a>
#### public $objectType : string
---
***Description***

Тип объекта: приход "payment" или возврат "refund".

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**


<a name="property_on_behalf_of"></a>
#### public $on_behalf_of : string
---
***Description***

Идентификатор магазина в ЮKassa.

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**


<a name="property_onBehalfOf"></a>
#### public $onBehalfOf : string
---
***Description***

Идентификатор магазина в ЮKassa.

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**


<a name="property_receipt_industry_details"></a>
#### public $receipt_industry_details : array
---
***Description***

Отраслевой реквизит предмета расчета.

**Type:** <a href="../array"><abbr title="array">array</abbr></a>

**Details:**


<a name="property_receipt_operational_details"></a>
#### public $receipt_operational_details : \YooKassa\Model\Receipt\OperationalDetails
---
***Description***

Операционный реквизит чека.

**Type:** <a href="../classes/YooKassa-Model-Receipt-OperationalDetails.html"><abbr title="\YooKassa\Model\Receipt\OperationalDetails">OperationalDetails</abbr></a>

**Details:**


<a name="property_receiptIndustryDetails"></a>
#### public $receiptIndustryDetails : array
---
***Description***

Отраслевой реквизит предмета расчета.

**Type:** <a href="../array"><abbr title="array">array</abbr></a>

**Details:**


<a name="property_receiptOperationalDetails"></a>
#### public $receiptOperationalDetails : \YooKassa\Model\Receipt\OperationalDetails
---
***Description***

Операционный реквизит чека.

**Type:** <a href="../classes/YooKassa-Model-Receipt-OperationalDetails.html"><abbr title="\YooKassa\Model\Receipt\OperationalDetails">OperationalDetails</abbr></a>

**Details:**


<a name="property_send"></a>
#### public $send : bool
---
***Description***

Формирование чека в онлайн-кассе сразу после создания объекта чека. Сейчас можно передать только значение ~`true`.

**Type:** <a href="../bool"><abbr title="bool">bool</abbr></a>

**Details:**


<a name="property_settlements"></a>
#### public $settlements : array
---
***Description***

Список платежей

**Type:** <a href="../array"><abbr title="array">array</abbr></a>

**Details:**


<a name="property_tax_system_code"></a>
#### public $tax_system_code : int
---
***Description***

Система налогообложения магазина (тег в 54 ФЗ — 1055).

**Type:** <a href="../int"><abbr title="int">int</abbr></a>

**Details:**


<a name="property_taxSystemCode"></a>
#### public $taxSystemCode : int
---
***Description***

Система налогообложения магазина (тег в 54 ФЗ — 1055).

**Type:** <a href="../int"><abbr title="int">int</abbr></a>

**Details:**


<a name="property_type"></a>
#### public $type : string
---
***Description***

Тип чека в онлайн-кассе: приход "payment" или возврат "refund".

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

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


<a name="method_builder" class="anchor"></a>
#### public builder() : \YooKassa\Request\Receipts\CreatePostReceiptRequestBuilder

```php
Static public builder() : \YooKassa\Request\Receipts\CreatePostReceiptRequestBuilder
```

**Summary**

Возвращает билдер объектов запросов создания платежа.

**Details:**
* Inherited From: [\YooKassa\Request\Receipts\CreatePostReceiptRequest](../classes/YooKassa-Request-Receipts-CreatePostReceiptRequest.md)

**Returns:** \YooKassa\Request\Receipts\CreatePostReceiptRequestBuilder - Инстанс билдера объектов запросов


<a name="method_clearValidationError" class="anchor"></a>
#### public clearValidationError() : void

```php
public clearValidationError() : void
```

**Summary**

Очищает статус валидации текущего запроса.

**Details:**
* Inherited From: [\YooKassa\Common\AbstractRequest](../classes/YooKassa-Common-AbstractRequest.md)

**Returns:** void - 


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
* Inherited From: [\YooKassa\Request\Receipts\CreatePostReceiptRequest](../classes/YooKassa-Request-Receipts-CreatePostReceiptRequest.md)

**Returns:** \YooKassa\Model\Receipt\AdditionalUserProps|null - Дополнительный реквизит пользователя


<a name="method_getCustomer" class="anchor"></a>
#### public getCustomer() : \YooKassa\Model\Receipt\ReceiptCustomerInterface|null

```php
public getCustomer() : \YooKassa\Model\Receipt\ReceiptCustomerInterface|null
```

**Summary**

Возвращает информацию о плательщике.

**Details:**
* Inherited From: [\YooKassa\Request\Receipts\CreatePostReceiptRequest](../classes/YooKassa-Request-Receipts-CreatePostReceiptRequest.md)

**Returns:** \YooKassa\Model\Receipt\ReceiptCustomerInterface|null - Информация о плательщике


<a name="method_getItems" class="anchor"></a>
#### public getItems() : \YooKassa\Model\Receipt\ReceiptItemInterface[]|\YooKassa\Common\ListObjectInterface

```php
public getItems() : \YooKassa\Model\Receipt\ReceiptItemInterface[]|\YooKassa\Common\ListObjectInterface
```

**Summary**

Возвращает список позиций в текущем чеке.

**Details:**
* Inherited From: [\YooKassa\Request\Receipts\CreatePostReceiptRequest](../classes/YooKassa-Request-Receipts-CreatePostReceiptRequest.md)

**Returns:** \YooKassa\Model\Receipt\ReceiptItemInterface[]|\YooKassa\Common\ListObjectInterface - Список товаров в заказе


<a name="method_getLastValidationError" class="anchor"></a>
#### public getLastValidationError() : string|null

```php
public getLastValidationError() : string|null
```

**Summary**

Возвращает последнюю ошибку валидации.

**Details:**
* Inherited From: [\YooKassa\Common\AbstractRequest](../classes/YooKassa-Common-AbstractRequest.md)

**Returns:** string|null - Последняя произошедшая ошибка валидации


<a name="method_getObjectId" class="anchor"></a>
#### public getObjectId() : string|null

```php
public getObjectId() : string|null
```

**Summary**

Возвращает Id объекта чека.

**Details:**
* Inherited From: [\YooKassa\Request\Receipts\CreatePostReceiptRequest](../classes/YooKassa-Request-Receipts-CreatePostReceiptRequest.md)

**Returns:** string|null - Id объекта чека


<a name="method_getObjectType" class="anchor"></a>
#### public getObjectType() : string|null

```php
public getObjectType() : string|null
```

**Summary**

Возвращает тип объекта чека.

**Details:**
* Inherited From: [\YooKassa\Request\Receipts\CreatePostReceiptRequest](../classes/YooKassa-Request-Receipts-CreatePostReceiptRequest.md)

**Returns:** string|null - Тип объекта чека


<a name="method_getOnBehalfOf" class="anchor"></a>
#### public getOnBehalfOf() : string|null

```php
public getOnBehalfOf() : string|null
```

**Summary**

Возвращает идентификатор магазина, от имени которого нужно отправить чек.

**Description**

Выдается ЮKassa, отображается в разделе Продавцы личного кабинета (столбец shopId).
Необходимо передавать, если вы используете решение ЮKassa для платформ.

**Details:**
* Inherited From: [\YooKassa\Request\Receipts\CreatePostReceiptRequest](../classes/YooKassa-Request-Receipts-CreatePostReceiptRequest.md)

**Returns:** string|null - 


<a name="method_getReceiptIndustryDetails" class="anchor"></a>
#### public getReceiptIndustryDetails() : \YooKassa\Model\Receipt\IndustryDetails[]|\YooKassa\Common\ListObjectInterface

```php
public getReceiptIndustryDetails() : \YooKassa\Model\Receipt\IndustryDetails[]|\YooKassa\Common\ListObjectInterface
```

**Summary**

Возвращает отраслевой реквизит чека.

**Details:**
* Inherited From: [\YooKassa\Request\Receipts\CreatePostReceiptRequest](../classes/YooKassa-Request-Receipts-CreatePostReceiptRequest.md)

**Returns:** \YooKassa\Model\Receipt\IndustryDetails[]|\YooKassa\Common\ListObjectInterface - Отраслевой реквизит чека


<a name="method_getReceiptOperationalDetails" class="anchor"></a>
#### public getReceiptOperationalDetails() : \YooKassa\Model\Receipt\OperationalDetails|null

```php
public getReceiptOperationalDetails() : \YooKassa\Model\Receipt\OperationalDetails|null
```

**Summary**

Возвращает операционный реквизит чека.

**Details:**
* Inherited From: [\YooKassa\Request\Receipts\CreatePostReceiptRequest](../classes/YooKassa-Request-Receipts-CreatePostReceiptRequest.md)

**Returns:** \YooKassa\Model\Receipt\OperationalDetails|null - Операционный реквизит чека


<a name="method_getSend" class="anchor"></a>
#### public getSend() : bool

```php
public getSend() : bool
```

**Summary**

Возвращает признак отложенной отправки чека.

**Details:**
* Inherited From: [\YooKassa\Request\Receipts\CreatePostReceiptRequest](../classes/YooKassa-Request-Receipts-CreatePostReceiptRequest.md)

**Returns:** bool - Признак отложенной отправки чека


<a name="method_getSettlements" class="anchor"></a>
#### public getSettlements() : \YooKassa\Model\Receipt\SettlementInterface[]|\YooKassa\Common\ListObjectInterface

```php
public getSettlements() : \YooKassa\Model\Receipt\SettlementInterface[]|\YooKassa\Common\ListObjectInterface
```

**Summary**

Возвращает массив оплат, обеспечивающих выдачу товара.

**Details:**
* Inherited From: [\YooKassa\Request\Receipts\CreatePostReceiptRequest](../classes/YooKassa-Request-Receipts-CreatePostReceiptRequest.md)

**Returns:** \YooKassa\Model\Receipt\SettlementInterface[]|\YooKassa\Common\ListObjectInterface - Массив оплат, обеспечивающих выдачу товара


<a name="method_getTaxSystemCode" class="anchor"></a>
#### public getTaxSystemCode() : int|null

```php
public getTaxSystemCode() : int|null
```

**Summary**

Возвращает код системы налогообложения.

**Details:**
* Inherited From: [\YooKassa\Request\Receipts\CreatePostReceiptRequest](../classes/YooKassa-Request-Receipts-CreatePostReceiptRequest.md)

**Returns:** int|null - Код системы налогообложения. Число 1-6.


<a name="method_getType" class="anchor"></a>
#### public getType() : string|null

```php
public getType() : string|null
```

**Summary**

Возвращает тип чека в онлайн-кассе.

**Details:**
* Inherited From: [\YooKassa\Request\Receipts\CreatePostReceiptRequest](../classes/YooKassa-Request-Receipts-CreatePostReceiptRequest.md)

**Returns:** string|null - Тип чека в онлайн-кассе: приход "payment" или возврат "refund"


<a name="method_getValidator" class="anchor"></a>
#### public getValidator() : \YooKassa\Validator\Validator

```php
public getValidator() : \YooKassa\Validator\Validator
```

**Details:**
* Inherited From: [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)

**Returns:** \YooKassa\Validator\Validator - 


<a name="method_hasCustomer" class="anchor"></a>
#### public hasCustomer() : bool

```php
public hasCustomer() : bool
```

**Summary**

Проверяет наличие данных о плательщике.

**Details:**
* Inherited From: [\YooKassa\Request\Receipts\CreatePostReceiptRequest](../classes/YooKassa-Request-Receipts-CreatePostReceiptRequest.md)

**Returns:** bool - 


<a name="method_jsonSerialize" class="anchor"></a>
#### public jsonSerialize() : array

```php
public jsonSerialize() : array
```

**Summary**

Возвращает ассоциативный массив со свойствами текущего объекта для его дальнейшей JSON сериализации.

**Details:**
* Inherited From: [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)

**Returns:** array - Ассоциативный массив со свойствами текущего объекта


<a name="method_notEmpty" class="anchor"></a>
#### public notEmpty() : bool

```php
public notEmpty() : bool
```

**Summary**

Проверяет есть ли в чеке хотя бы одна позиция товаров и оплат

**Details:**
* Inherited From: [\YooKassa\Request\Receipts\CreatePostReceiptRequest](../classes/YooKassa-Request-Receipts-CreatePostReceiptRequest.md)

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


<a name="method_setAdditionalUserProps" class="anchor"></a>
#### public setAdditionalUserProps() : self

```php
public setAdditionalUserProps(null|\YooKassa\Model\Receipt\AdditionalUserProps|array $additional_user_props = null) : self
```

**Summary**

Устанавливает дополнительный реквизит пользователя.

**Details:**
* Inherited From: [\YooKassa\Request\Receipts\CreatePostReceiptRequest](../classes/YooKassa-Request-Receipts-CreatePostReceiptRequest.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">null OR \YooKassa\Model\Receipt\AdditionalUserProps OR array</code> | additional_user_props  | Дополнительный реквизит пользователя |

**Returns:** self - 


<a name="method_setCustomer" class="anchor"></a>
#### public setCustomer() : self

```php
public setCustomer(array|\YooKassa\Model\Receipt\ReceiptCustomerInterface $customer = null) : self
```

**Summary**

Устанавливает информацию о плательщике.

**Details:**
* Inherited From: [\YooKassa\Request\Receipts\CreatePostReceiptRequest](../classes/YooKassa-Request-Receipts-CreatePostReceiptRequest.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">array OR \YooKassa\Model\Receipt\ReceiptCustomerInterface</code> | customer  | Информация о плательщике |

**Returns:** self - 


<a name="method_setItems" class="anchor"></a>
#### public setItems() : self

```php
public setItems(array|\YooKassa\Common\ListObjectInterface $items = null) : self
```

**Summary**

Устанавливает список позиций в чеке.

**Description**

Если до этого в чеке уже были установлены значения, они удаляются и полностью заменяются переданным списком
позиций. Все передаваемые значения в массиве позиций должны быть объектами класса, реализующего интерфейс
ReceiptItemInterface, в противном случае будет выброшено исключение InvalidPropertyValueTypeException.

**Details:**
* Inherited From: [\YooKassa\Request\Receipts\CreatePostReceiptRequest](../classes/YooKassa-Request-Receipts-CreatePostReceiptRequest.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">array OR \YooKassa\Common\ListObjectInterface</code> | items  | Список товаров в заказе: для [Чеков от ЮKassa](https://yookassa.ru/developers/payment-acceptance/receipts/54fz/yoomoney/basics) — не более 80 товаров, для [сторонних онлайн-касс](https://yookassa.ru/developers/payment-acceptance/receipts/54fz/other-services/basics) — не более 100 товаров. |

**Returns:** self - 


<a name="method_setObjectId" class="anchor"></a>
#### public setObjectId() : \YooKassa\Request\Receipts\CreatePostReceiptRequest

```php
public setObjectId(string|null $value) : \YooKassa\Request\Receipts\CreatePostReceiptRequest
```

**Summary**

Устанавливает Id объекта чека.

**Details:**
* Inherited From: [\YooKassa\Request\Receipts\CreatePostReceiptRequest](../classes/YooKassa-Request-Receipts-CreatePostReceiptRequest.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | value  | Id объекта чека |

**Returns:** \YooKassa\Request\Receipts\CreatePostReceiptRequest - 


<a name="method_setObjectType" class="anchor"></a>
#### public setObjectType() : \YooKassa\Request\Receipts\CreatePostReceiptRequest

```php
public setObjectType(string|null $value) : \YooKassa\Request\Receipts\CreatePostReceiptRequest
```

**Summary**

Устанавливает тип объекта чека.

**Details:**
* Inherited From: [\YooKassa\Request\Receipts\CreatePostReceiptRequest](../classes/YooKassa-Request-Receipts-CreatePostReceiptRequest.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | value  | Тип объекта чека |

**Returns:** \YooKassa\Request\Receipts\CreatePostReceiptRequest - 


<a name="method_setOnBehalfOf" class="anchor"></a>
#### public setOnBehalfOf() : self

```php
public setOnBehalfOf(string|null $on_behalf_of = null) : self
```

**Summary**

Устанавливает идентификатор магазина, от имени которого нужно отправить чек.

**Description**

Выдается ЮKassa, отображается в разделе Продавцы личного кабинета (столбец shopId).
Необходимо передавать, если вы используете решение ЮKassa для платформ.

**Details:**
* Inherited From: [\YooKassa\Request\Receipts\CreatePostReceiptRequest](../classes/YooKassa-Request-Receipts-CreatePostReceiptRequest.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | on_behalf_of  |  |

**Returns:** self - 


<a name="method_setReceiptIndustryDetails" class="anchor"></a>
#### public setReceiptIndustryDetails() : self

```php
public setReceiptIndustryDetails(null|array|\YooKassa\Common\ListObjectInterface $receipt_industry_details = null) : self
```

**Summary**

Устанавливает отраслевой реквизит чека.

**Details:**
* Inherited From: [\YooKassa\Request\Receipts\CreatePostReceiptRequest](../classes/YooKassa-Request-Receipts-CreatePostReceiptRequest.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">null OR array OR \YooKassa\Common\ListObjectInterface</code> | receipt_industry_details  | Отраслевой реквизит чека |

**Returns:** self - 


<a name="method_setReceiptOperationalDetails" class="anchor"></a>
#### public setReceiptOperationalDetails() : self

```php
public setReceiptOperationalDetails(array|\YooKassa\Model\Receipt\OperationalDetails|null $receipt_operational_details = null) : self
```

**Summary**

Устанавливает операционный реквизит чека.

**Details:**
* Inherited From: [\YooKassa\Request\Receipts\CreatePostReceiptRequest](../classes/YooKassa-Request-Receipts-CreatePostReceiptRequest.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">array OR \YooKassa\Model\Receipt\OperationalDetails OR null</code> | receipt_operational_details  | Операционный реквизит чека |

**Returns:** self - 


<a name="method_setSend" class="anchor"></a>
#### public setSend() : self

```php
public setSend(bool $send = null) : self
```

**Summary**

Устанавливает признак отложенной отправки чека.

**Details:**
* Inherited From: [\YooKassa\Request\Receipts\CreatePostReceiptRequest](../classes/YooKassa-Request-Receipts-CreatePostReceiptRequest.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">bool</code> | send  | Признак отложенной отправки чека |

**Returns:** self - 


<a name="method_setSettlements" class="anchor"></a>
#### public setSettlements() : self

```php
public setSettlements(array|\YooKassa\Common\ListObjectInterface|null $settlements = null) : self
```

**Summary**

Устанавливает массив оплат, обеспечивающих выдачу товара.

**Details:**
* Inherited From: [\YooKassa\Request\Receipts\CreatePostReceiptRequest](../classes/YooKassa-Request-Receipts-CreatePostReceiptRequest.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">array OR \YooKassa\Common\ListObjectInterface OR null</code> | settlements  | Массив оплат, обеспечивающих выдачу товара |

**Returns:** self - 


<a name="method_setTaxSystemCode" class="anchor"></a>
#### public setTaxSystemCode() : self

```php
public setTaxSystemCode(int|null $tax_system_code = null) : self
```

**Summary**

Устанавливает код системы налогообложения.

**Details:**
* Inherited From: [\YooKassa\Request\Receipts\CreatePostReceiptRequest](../classes/YooKassa-Request-Receipts-CreatePostReceiptRequest.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">int OR null</code> | tax_system_code  | Код системы налогообложения. Число 1-6 |

**Returns:** self - 


<a name="method_setType" class="anchor"></a>
#### public setType() : self

```php
public setType(string|null $type = null) : self
```

**Summary**

Устанавливает тип чека в онлайн-кассе.

**Details:**
* Inherited From: [\YooKassa\Request\Receipts\CreatePostReceiptRequest](../classes/YooKassa-Request-Receipts-CreatePostReceiptRequest.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | type  | тип чека в онлайн-кассе: приход "payment" или возврат "refund" |

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


<a name="method_validate" class="anchor"></a>
#### public validate() : bool

```php
public validate() : bool
```

**Summary**

Валидирует текущий запрос, проверяет все ли нужные свойства установлены.

**Details:**
* Inherited From: [\YooKassa\Request\Receipts\CreatePostReceiptRequest](../classes/YooKassa-Request-Receipts-CreatePostReceiptRequest.md)

**Returns:** bool - True если запрос валиден, false если нет


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


<a name="method_setValidationError" class="anchor"></a>
#### protected setValidationError() : void

```php
protected setValidationError(string $value) : void
```

**Summary**

Устанавливает ошибку валидации.

**Details:**
* Inherited From: [\YooKassa\Common\AbstractRequest](../classes/YooKassa-Common-AbstractRequest.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string</code> | value  | Ошибка, произошедшая при валидации объекта |

**Returns:** void - 


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