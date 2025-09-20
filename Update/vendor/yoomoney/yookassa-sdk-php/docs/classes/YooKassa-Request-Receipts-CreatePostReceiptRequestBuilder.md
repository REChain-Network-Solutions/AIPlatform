# [YooKassa API SDK](../home.md)

# Class: \YooKassa\Request\Receipts\CreatePostReceiptRequestBuilder
### Namespace: [\YooKassa\Request\Receipts](../namespaces/yookassa-request-receipts.md)
---
**Summary:**

Класс билдера объектов запросов к API на создание чека.


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
| protected | [$currentObject](../classes/YooKassa-Request-Receipts-CreatePostReceiptRequestBuilder.md#property_currentObject) |  | Собираемый объект запроса. |

---
### Methods
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| public | [__construct()](../classes/YooKassa-Common-AbstractRequestBuilder.md#method___construct) |  | Конструктор, инициализирует пустой запрос, который в будущем начнём собирать. |
| public | [addItem()](../classes/YooKassa-Request-Receipts-CreatePostReceiptRequestBuilder.md#method_addItem) |  | Добавляет товар в чек. |
| public | [addReceiptIndustryDetails()](../classes/YooKassa-Request-Receipts-CreatePostReceiptRequestBuilder.md#method_addReceiptIndustryDetails) |  | Добавляет отраслевой реквизит чека. |
| public | [addSettlement()](../classes/YooKassa-Request-Receipts-CreatePostReceiptRequestBuilder.md#method_addSettlement) |  | Добавляет оплату в перечень совершенных расчетов. |
| public | [build()](../classes/YooKassa-Request-Receipts-CreatePostReceiptRequestBuilder.md#method_build) |  | Строит и возвращает объект запроса для отправки в API ЮKassa. |
| public | [setAdditionalUserProps()](../classes/YooKassa-Request-Receipts-CreatePostReceiptRequestBuilder.md#method_setAdditionalUserProps) |  | Устанавливает дополнительный реквизит пользователя. |
| public | [setCurrency()](../classes/YooKassa-Request-Receipts-CreatePostReceiptRequestBuilder.md#method_setCurrency) |  | Устанавливает валюту в которой будет происходить подтверждение оплаты заказа. |
| public | [setCustomer()](../classes/YooKassa-Request-Receipts-CreatePostReceiptRequestBuilder.md#method_setCustomer) |  | Устанавливает информацию о пользователе. |
| public | [setItems()](../classes/YooKassa-Request-Receipts-CreatePostReceiptRequestBuilder.md#method_setItems) |  | Устанавливает список товаров чека. |
| public | [setObjectId()](../classes/YooKassa-Request-Receipts-CreatePostReceiptRequestBuilder.md#method_setObjectId) |  | Устанавливает Id объекта чека. |
| public | [setObjectType()](../classes/YooKassa-Request-Receipts-CreatePostReceiptRequestBuilder.md#method_setObjectType) |  | Устанавливает тип объекта чека. |
| public | [setOnBehalfOf()](../classes/YooKassa-Request-Receipts-CreatePostReceiptRequestBuilder.md#method_setOnBehalfOf) |  | Устанавливает идентификатор магазина, от имени которого нужно отправить чек. |
| public | [setOptions()](../classes/YooKassa-Common-AbstractRequestBuilder.md#method_setOptions) |  | Устанавливает свойства запроса из массива. |
| public | [setReceiptIndustryDetails()](../classes/YooKassa-Request-Receipts-CreatePostReceiptRequestBuilder.md#method_setReceiptIndustryDetails) |  | Устанавливает отраслевой реквизит чека. |
| public | [setReceiptOperationalDetails()](../classes/YooKassa-Request-Receipts-CreatePostReceiptRequestBuilder.md#method_setReceiptOperationalDetails) |  | Устанавливает операционный реквизит чека. |
| public | [setSend()](../classes/YooKassa-Request-Receipts-CreatePostReceiptRequestBuilder.md#method_setSend) |  | Устанавливает признак отложенной отправки чека. |
| public | [setSettlements()](../classes/YooKassa-Request-Receipts-CreatePostReceiptRequestBuilder.md#method_setSettlements) |  | Устанавливает массив оплат, обеспечивающих выдачу товара. |
| public | [setTaxSystemCode()](../classes/YooKassa-Request-Receipts-CreatePostReceiptRequestBuilder.md#method_setTaxSystemCode) |  | Устанавливает код системы налогообложения. |
| public | [setType()](../classes/YooKassa-Request-Receipts-CreatePostReceiptRequestBuilder.md#method_setType) |  | Устанавливает тип чека в онлайн-кассе. |
| protected | [initCurrentObject()](../classes/YooKassa-Request-Receipts-CreatePostReceiptRequestBuilder.md#method_initCurrentObject) |  | Инициализирует объект запроса, который в дальнейшем будет собираться билдером |

---
### Details
* File: [lib/Request/Receipts/CreatePostReceiptRequestBuilder.php](../../lib/Request/Receipts/CreatePostReceiptRequestBuilder.php)
* Package: Default
* Class Hierarchy: 
  * [\YooKassa\Common\AbstractRequestBuilder](../classes/YooKassa-Common-AbstractRequestBuilder.md)
  * \YooKassa\Request\Receipts\CreatePostReceiptRequestBuilder

---
## Properties
<a name="property_currentObject"></a>
#### protected $currentObject : ?\YooKassa\Common\AbstractRequestInterface
---
**Summary**

Собираемый объект запроса.

**Type:** <a href="../?\YooKassa\Common\AbstractRequestInterface"><abbr title="?\YooKassa\Common\AbstractRequestInterface">AbstractRequestInterface</abbr></a>

**Details:**



---
## Methods
<a name="method___construct" class="anchor"></a>
#### public __construct() : mixed

```php
public __construct() : mixed
```

**Summary**

Конструктор, инициализирует пустой запрос, который в будущем начнём собирать.

**Details:**
* Inherited From: [\YooKassa\Common\AbstractRequestBuilder](../classes/YooKassa-Common-AbstractRequestBuilder.md)

**Returns:** mixed - 


<a name="method_addItem" class="anchor"></a>
#### public addItem() : self

```php
public addItem(array|\YooKassa\Model\Receipt\ReceiptItemInterface|null $value) : self
```

**Summary**

Добавляет товар в чек.

**Details:**
* Inherited From: [\YooKassa\Request\Receipts\CreatePostReceiptRequestBuilder](../classes/YooKassa-Request-Receipts-CreatePostReceiptRequestBuilder.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">array OR \YooKassa\Model\Receipt\ReceiptItemInterface OR null</code> | value  | Информация о товаре |

**Returns:** self - Инстанс билдера запросов


<a name="method_addReceiptIndustryDetails" class="anchor"></a>
#### public addReceiptIndustryDetails() : self

```php
public addReceiptIndustryDetails(array|\YooKassa\Model\Receipt\IndustryDetails|null $value) : self
```

**Summary**

Добавляет отраслевой реквизит чека.

**Details:**
* Inherited From: [\YooKassa\Request\Receipts\CreatePostReceiptRequestBuilder](../classes/YooKassa-Request-Receipts-CreatePostReceiptRequestBuilder.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">array OR \YooKassa\Model\Receipt\IndustryDetails OR null</code> | value  | Отраслевой реквизит чека |

**Returns:** self - Инстанс билдера запросов


<a name="method_addSettlement" class="anchor"></a>
#### public addSettlement() : self

```php
public addSettlement(array|\YooKassa\Model\Receipt\SettlementInterface|null $value) : self
```

**Summary**

Добавляет оплату в перечень совершенных расчетов.

**Details:**
* Inherited From: [\YooKassa\Request\Receipts\CreatePostReceiptRequestBuilder](../classes/YooKassa-Request-Receipts-CreatePostReceiptRequestBuilder.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">array OR \YooKassa\Model\Receipt\SettlementInterface OR null</code> | value  | Информация об оплате |

**Returns:** self - Инстанс билдера запросов


<a name="method_build" class="anchor"></a>
#### public build() : \YooKassa\Request\Receipts\CreatePostReceiptRequest

```php
public build(null|array $options = null) : \YooKassa\Request\Receipts\CreatePostReceiptRequest
```

**Summary**

Строит и возвращает объект запроса для отправки в API ЮKassa.

**Details:**
* Inherited From: [\YooKassa\Request\Receipts\CreatePostReceiptRequestBuilder](../classes/YooKassa-Request-Receipts-CreatePostReceiptRequestBuilder.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">null OR array</code> | options  | Массив параметров для установки в объект запроса |

##### Throws:
| Type | Description |
| ---- | ----------- |
| \YooKassa\Common\Exceptions\InvalidRequestException | Выбрасывается если собрать объект запроса не удалось |

**Returns:** \YooKassa\Request\Receipts\CreatePostReceiptRequest - Инстанс объекта запроса


<a name="method_setAdditionalUserProps" class="anchor"></a>
#### public setAdditionalUserProps() : self

```php
public setAdditionalUserProps(\YooKassa\Model\Receipt\AdditionalUserProps|array|null $value) : self
```

**Summary**

Устанавливает дополнительный реквизит пользователя.

**Details:**
* Inherited From: [\YooKassa\Request\Receipts\CreatePostReceiptRequestBuilder](../classes/YooKassa-Request-Receipts-CreatePostReceiptRequestBuilder.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">\YooKassa\Model\Receipt\AdditionalUserProps OR array OR null</code> | value  | Дополнительный реквизит пользователя |

**Returns:** self - Инстанс билдера запросов


<a name="method_setCurrency" class="anchor"></a>
#### public setCurrency() : self

```php
public setCurrency(string $value) : self
```

**Summary**

Устанавливает валюту в которой будет происходить подтверждение оплаты заказа.

**Details:**
* Inherited From: [\YooKassa\Request\Receipts\CreatePostReceiptRequestBuilder](../classes/YooKassa-Request-Receipts-CreatePostReceiptRequestBuilder.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string</code> | value  | Валюта в которой подтверждается оплата |

**Returns:** self - Инстанс билдера запросов


<a name="method_setCustomer" class="anchor"></a>
#### public setCustomer() : self

```php
public setCustomer(array|\YooKassa\Model\Receipt\ReceiptCustomerInterface $value) : self
```

**Summary**

Устанавливает информацию о пользователе.

**Details:**
* Inherited From: [\YooKassa\Request\Receipts\CreatePostReceiptRequestBuilder](../classes/YooKassa-Request-Receipts-CreatePostReceiptRequestBuilder.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">array OR \YooKassa\Model\Receipt\ReceiptCustomerInterface</code> | value  | Информация о плательщике |

**Returns:** self - Инстанс билдера запросов


<a name="method_setItems" class="anchor"></a>
#### public setItems() : self

```php
public setItems(array|\YooKassa\Model\Receipt\ReceiptItemInterface[]|null $value) : self
```

**Summary**

Устанавливает список товаров чека.

**Details:**
* Inherited From: [\YooKassa\Request\Receipts\CreatePostReceiptRequestBuilder](../classes/YooKassa-Request-Receipts-CreatePostReceiptRequestBuilder.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">array OR \YooKassa\Model\Receipt\ReceiptItemInterface[] OR null</code> | value  | Список товаров чека |

**Returns:** self - Инстанс билдера запросов


<a name="method_setObjectId" class="anchor"></a>
#### public setObjectId() : self

```php
public setObjectId(string $value, null|string $type = null) : self
```

**Summary**

Устанавливает Id объекта чека.

**Details:**
* Inherited From: [\YooKassa\Request\Receipts\CreatePostReceiptRequestBuilder](../classes/YooKassa-Request-Receipts-CreatePostReceiptRequestBuilder.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string</code> | value  | Id объекта чека |
| <code lang="php">null OR string</code> | type  | Тип объекта чека |

**Returns:** self - Инстанс билдера запросов


<a name="method_setObjectType" class="anchor"></a>
#### public setObjectType() : self

```php
public setObjectType(string|null $value) : self
```

**Summary**

Устанавливает тип объекта чека.

**Details:**
* Inherited From: [\YooKassa\Request\Receipts\CreatePostReceiptRequestBuilder](../classes/YooKassa-Request-Receipts-CreatePostReceiptRequestBuilder.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | value  | Тип объекта чека |

**Returns:** self - Инстанс билдера запросов


<a name="method_setOnBehalfOf" class="anchor"></a>
#### public setOnBehalfOf() : self

```php
public setOnBehalfOf(string|null $value) : self
```

**Summary**

Устанавливает идентификатор магазина, от имени которого нужно отправить чек.

**Description**

Выдается ЮKassa, отображается в разделе Продавцы личного кабинета (столбец shopId).
Необходимо передавать, если вы используете решение ЮKassa для платформ.

**Details:**
* Inherited From: [\YooKassa\Request\Receipts\CreatePostReceiptRequestBuilder](../classes/YooKassa-Request-Receipts-CreatePostReceiptRequestBuilder.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | value  | Идентификатор магазина, от имени которого нужно отправить чек |

**Returns:** self - Инстанс билдера запросов


<a name="method_setOptions" class="anchor"></a>
#### public setOptions() : \YooKassa\Common\AbstractRequestBuilder

```php
public setOptions(iterable|null $options) : \YooKassa\Common\AbstractRequestBuilder
```

**Summary**

Устанавливает свойства запроса из массива.

**Details:**
* Inherited From: [\YooKassa\Common\AbstractRequestBuilder](../classes/YooKassa-Common-AbstractRequestBuilder.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">iterable OR null</code> | options  | Массив свойств запроса |

##### Throws:
| Type | Description |
| ---- | ----------- |
| \InvalidArgumentException | Выбрасывается если аргумент не массив и не итерируемый объект |
| \YooKassa\Common\Exceptions\InvalidPropertyException | Выбрасывается если не удалось установить один из параметров, переданных в массиве настроек |

**Returns:** \YooKassa\Common\AbstractRequestBuilder - Инстанс текущего билдера запросов


<a name="method_setReceiptIndustryDetails" class="anchor"></a>
#### public setReceiptIndustryDetails() : self

```php
public setReceiptIndustryDetails(array|\YooKassa\Model\Receipt\IndustryDetails[]|null $value) : self
```

**Summary**

Устанавливает отраслевой реквизит чека.

**Details:**
* Inherited From: [\YooKassa\Request\Receipts\CreatePostReceiptRequestBuilder](../classes/YooKassa-Request-Receipts-CreatePostReceiptRequestBuilder.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">array OR \YooKassa\Model\Receipt\IndustryDetails[] OR null</code> | value  | Отраслевой реквизит чека |

**Returns:** self - Инстанс билдера запросов


<a name="method_setReceiptOperationalDetails" class="anchor"></a>
#### public setReceiptOperationalDetails() : self

```php
public setReceiptOperationalDetails(array|\YooKassa\Model\Receipt\OperationalDetails|null $value) : self
```

**Summary**

Устанавливает операционный реквизит чека.

**Details:**
* Inherited From: [\YooKassa\Request\Receipts\CreatePostReceiptRequestBuilder](../classes/YooKassa-Request-Receipts-CreatePostReceiptRequestBuilder.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">array OR \YooKassa\Model\Receipt\OperationalDetails OR null</code> | value  | Операционный реквизит чека |

**Returns:** self - Инстанс билдера запросов


<a name="method_setSend" class="anchor"></a>
#### public setSend() : self

```php
public setSend(bool $value) : self
```

**Summary**

Устанавливает признак отложенной отправки чека.

**Details:**
* Inherited From: [\YooKassa\Request\Receipts\CreatePostReceiptRequestBuilder](../classes/YooKassa-Request-Receipts-CreatePostReceiptRequestBuilder.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">bool</code> | value  | Признак отложенной отправки чека |

**Returns:** self - Инстанс билдера запросов


<a name="method_setSettlements" class="anchor"></a>
#### public setSettlements() : self

```php
public setSettlements(array|\YooKassa\Model\Receipt\SettlementInterface[]|null $value) : self
```

**Summary**

Устанавливает массив оплат, обеспечивающих выдачу товара.

**Details:**
* Inherited From: [\YooKassa\Request\Receipts\CreatePostReceiptRequestBuilder](../classes/YooKassa-Request-Receipts-CreatePostReceiptRequestBuilder.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">array OR \YooKassa\Model\Receipt\SettlementInterface[] OR null</code> | value  | Массив оплат, обеспечивающих выдачу товара |

**Returns:** self - Инстанс билдера запросов


<a name="method_setTaxSystemCode" class="anchor"></a>
#### public setTaxSystemCode() : self

```php
public setTaxSystemCode(int|null $value) : self
```

**Summary**

Устанавливает код системы налогообложения.

**Details:**
* Inherited From: [\YooKassa\Request\Receipts\CreatePostReceiptRequestBuilder](../classes/YooKassa-Request-Receipts-CreatePostReceiptRequestBuilder.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">int OR null</code> | value  | Код системы налогообложения. Число 1-6. |

**Returns:** self - Инстанс билдера запросов


<a name="method_setType" class="anchor"></a>
#### public setType() : self

```php
public setType(string|null $value) : self
```

**Summary**

Устанавливает тип чека в онлайн-кассе.

**Details:**
* Inherited From: [\YooKassa\Request\Receipts\CreatePostReceiptRequestBuilder](../classes/YooKassa-Request-Receipts-CreatePostReceiptRequestBuilder.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | value  | Тип чека в онлайн-кассе: приход "payment" или возврат "refund" |

**Returns:** self - Инстанс билдера запросов


<a name="method_initCurrentObject" class="anchor"></a>
#### protected initCurrentObject() : \YooKassa\Request\Receipts\CreatePostReceiptRequest

```php
protected initCurrentObject() : \YooKassa\Request\Receipts\CreatePostReceiptRequest
```

**Summary**

Инициализирует объект запроса, который в дальнейшем будет собираться билдером

**Details:**
* Inherited From: [\YooKassa\Request\Receipts\CreatePostReceiptRequestBuilder](../classes/YooKassa-Request-Receipts-CreatePostReceiptRequestBuilder.md)

**Returns:** \YooKassa\Request\Receipts\CreatePostReceiptRequest - Инстанс собираемого объекта запроса к API



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