# [YooKassa API SDK](../home.md)

# Class: \YooKassa\Request\Receipts\ReceiptsRequest
### Namespace: [\YooKassa\Request\Receipts](../namespaces/yookassa-request-receipts.md)
---
**Summary:**

Класс объекта запроса к API списка возвратов магазина.


---
### Constants
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| public | [MAX_LIMIT_VALUE](../classes/YooKassa-Request-Receipts-ReceiptsRequest.md#constant_MAX_LIMIT_VALUE) |  | Максимальное количество объектов чеков в выборке |
| public | [LENGTH_PAYMENT_ID](../classes/YooKassa-Request-Receipts-ReceiptsRequest.md#constant_LENGTH_PAYMENT_ID) |  | Длина идентификатора платежа |
| public | [LENGTH_REFUND_ID](../classes/YooKassa-Request-Receipts-ReceiptsRequest.md#constant_LENGTH_REFUND_ID) |  | Длина идентификатора платежа |

---
### Properties
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| public | [$created_at_gt](../classes/YooKassa-Request-Receipts-ReceiptsRequest.md#property_created_at_gt) |  | Время создания, от (не включая) |
| public | [$created_at_gte](../classes/YooKassa-Request-Receipts-ReceiptsRequest.md#property_created_at_gte) |  | Время создания, от (включительно) |
| public | [$created_at_lt](../classes/YooKassa-Request-Receipts-ReceiptsRequest.md#property_created_at_lt) |  | Время создания, до (не включая) |
| public | [$created_at_lte](../classes/YooKassa-Request-Receipts-ReceiptsRequest.md#property_created_at_lte) |  | Время создания, до (включительно) |
| public | [$createdAtGt](../classes/YooKassa-Request-Receipts-ReceiptsRequest.md#property_createdAtGt) |  | Время создания, от (не включая) |
| public | [$createdAtGte](../classes/YooKassa-Request-Receipts-ReceiptsRequest.md#property_createdAtGte) |  | Время создания, от (включительно) |
| public | [$createdAtLt](../classes/YooKassa-Request-Receipts-ReceiptsRequest.md#property_createdAtLt) |  | Время создания, до (не включая) |
| public | [$createdAtLte](../classes/YooKassa-Request-Receipts-ReceiptsRequest.md#property_createdAtLte) |  | Время создания, до (включительно) |
| public | [$cursor](../classes/YooKassa-Request-Receipts-ReceiptsRequest.md#property_cursor) |  | Токен для получения следующей страницы выборки |
| public | [$limit](../classes/YooKassa-Request-Receipts-ReceiptsRequest.md#property_limit) |  | Ограничение количества объектов возврата, отображаемых на одной странице выдачи |
| public | [$payment_id](../classes/YooKassa-Request-Receipts-ReceiptsRequest.md#property_payment_id) |  | Идентификатор платежа |
| public | [$paymentId](../classes/YooKassa-Request-Receipts-ReceiptsRequest.md#property_paymentId) |  | Идентификатор платежа |
| public | [$refund_id](../classes/YooKassa-Request-Receipts-ReceiptsRequest.md#property_refund_id) |  | Идентификатор возврата |
| public | [$refundId](../classes/YooKassa-Request-Receipts-ReceiptsRequest.md#property_refundId) |  | Идентификатор возврата |
| public | [$status](../classes/YooKassa-Request-Receipts-ReceiptsRequest.md#property_status) |  | Статус возврата |

---
### Methods
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| public | [__construct()](../classes/YooKassa-Common-AbstractObject.md#method___construct) |  | AbstractObject constructor. |
| public | [__get()](../classes/YooKassa-Common-AbstractObject.md#method___get) |  | Возвращает значение свойства. |
| public | [__isset()](../classes/YooKassa-Common-AbstractObject.md#method___isset) |  | Проверяет наличие свойства. |
| public | [__set()](../classes/YooKassa-Common-AbstractObject.md#method___set) |  | Устанавливает значение свойства. |
| public | [__unset()](../classes/YooKassa-Common-AbstractObject.md#method___unset) |  | Удаляет свойство. |
| public | [builder()](../classes/YooKassa-Request-Receipts-ReceiptsRequest.md#method_builder) |  | Возвращает инстанс билдера объектов запросов списка возвратов магазина. |
| public | [clearValidationError()](../classes/YooKassa-Common-AbstractRequest.md#method_clearValidationError) |  | Очищает статус валидации текущего запроса. |
| public | [fromArray()](../classes/YooKassa-Common-AbstractObject.md#method_fromArray) |  | Устанавливает значения свойств текущего объекта из массива. |
| public | [getCreatedAtGt()](../classes/YooKassa-Request-Receipts-ReceiptsRequest.md#method_getCreatedAtGt) |  | Возвращает дату создания от которой будут возвращены возвраты или null, если дата не была установлена. |
| public | [getCreatedAtGte()](../classes/YooKassa-Request-Receipts-ReceiptsRequest.md#method_getCreatedAtGte) |  | Возвращает дату создания от которой будут возвращены возвраты или null, если дата не была установлена. |
| public | [getCreatedAtLt()](../classes/YooKassa-Request-Receipts-ReceiptsRequest.md#method_getCreatedAtLt) |  | Возвращает дату создания до которой будут возвращены возвраты или null, если дата не была установлена. |
| public | [getCreatedAtLte()](../classes/YooKassa-Request-Receipts-ReceiptsRequest.md#method_getCreatedAtLte) |  | Возвращает дату создания до которой будут возвращены возвраты или null, если дата не была установлена. |
| public | [getCursor()](../classes/YooKassa-Request-Receipts-ReceiptsRequest.md#method_getCursor) |  | Возвращает токен для получения следующей страницы выборки. |
| public | [getLastValidationError()](../classes/YooKassa-Common-AbstractRequest.md#method_getLastValidationError) |  | Возвращает последнюю ошибку валидации. |
| public | [getLimit()](../classes/YooKassa-Request-Receipts-ReceiptsRequest.md#method_getLimit) |  | Ограничение количества объектов платежа. |
| public | [getPaymentId()](../classes/YooKassa-Request-Receipts-ReceiptsRequest.md#method_getPaymentId) |  | Возвращает идентификатор платежа если он задан или null. |
| public | [getRefundId()](../classes/YooKassa-Request-Receipts-ReceiptsRequest.md#method_getRefundId) |  | Возвращает идентификатор возврата. |
| public | [getStatus()](../classes/YooKassa-Request-Receipts-ReceiptsRequest.md#method_getStatus) |  | Возвращает статус выбираемых возвратов или null, если он до этого не был установлен. |
| public | [getValidator()](../classes/YooKassa-Common-AbstractObject.md#method_getValidator) |  |  |
| public | [hasCreatedAtGt()](../classes/YooKassa-Request-Receipts-ReceiptsRequest.md#method_hasCreatedAtGt) |  | Проверяет, была ли установлена дата создания от которой выбираются возвраты. |
| public | [hasCreatedAtGte()](../classes/YooKassa-Request-Receipts-ReceiptsRequest.md#method_hasCreatedAtGte) |  | Проверяет, была ли установлена дата создания от которой выбираются возвраты. |
| public | [hasCreatedAtLt()](../classes/YooKassa-Request-Receipts-ReceiptsRequest.md#method_hasCreatedAtLt) |  | Проверяет, была ли установлена дата создания до которой выбираются возвраты. |
| public | [hasCreatedAtLte()](../classes/YooKassa-Request-Receipts-ReceiptsRequest.md#method_hasCreatedAtLte) |  | Проверяет, была ли установлена дата создания до которой выбираются возвраты. |
| public | [hasCursor()](../classes/YooKassa-Request-Receipts-ReceiptsRequest.md#method_hasCursor) |  | Проверяет, был ли установлен токен следующей страницы. |
| public | [hasLimit()](../classes/YooKassa-Request-Receipts-ReceiptsRequest.md#method_hasLimit) |  | Проверяет, было ли установлено ограничение количества объектов платежа. |
| public | [hasPaymentId()](../classes/YooKassa-Request-Receipts-ReceiptsRequest.md#method_hasPaymentId) |  | Проверяет, был ли задан идентификатор платежа. |
| public | [hasRefundId()](../classes/YooKassa-Request-Receipts-ReceiptsRequest.md#method_hasRefundId) |  | Проверяет, был ли установлен идентификатор возврата. |
| public | [hasStatus()](../classes/YooKassa-Request-Receipts-ReceiptsRequest.md#method_hasStatus) |  | Проверяет, был ли установлен статус выбираемых возвратов. |
| public | [jsonSerialize()](../classes/YooKassa-Common-AbstractObject.md#method_jsonSerialize) |  | Возвращает ассоциативный массив со свойствами текущего объекта для его дальнейшей JSON сериализации. |
| public | [offsetExists()](../classes/YooKassa-Common-AbstractObject.md#method_offsetExists) |  | Проверяет наличие свойства. |
| public | [offsetGet()](../classes/YooKassa-Common-AbstractObject.md#method_offsetGet) |  | Возвращает значение свойства. |
| public | [offsetSet()](../classes/YooKassa-Common-AbstractObject.md#method_offsetSet) |  | Устанавливает значение свойства. |
| public | [offsetUnset()](../classes/YooKassa-Common-AbstractObject.md#method_offsetUnset) |  | Удаляет свойство. |
| public | [setCreatedAtGt()](../classes/YooKassa-Request-Receipts-ReceiptsRequest.md#method_setCreatedAtGt) |  | Устанавливает дату создания от которой выбираются возвраты. |
| public | [setCreatedAtGte()](../classes/YooKassa-Request-Receipts-ReceiptsRequest.md#method_setCreatedAtGte) |  | Устанавливает дату создания от которой выбираются возвраты. |
| public | [setCreatedAtLt()](../classes/YooKassa-Request-Receipts-ReceiptsRequest.md#method_setCreatedAtLt) |  | Устанавливает дату создания до которой выбираются возвраты. |
| public | [setCreatedAtLte()](../classes/YooKassa-Request-Receipts-ReceiptsRequest.md#method_setCreatedAtLte) |  | Устанавливает дату создания до которой выбираются возвраты. |
| public | [setCursor()](../classes/YooKassa-Request-Receipts-ReceiptsRequest.md#method_setCursor) |  | Устанавливает токен следующей страницы выборки. |
| public | [setLimit()](../classes/YooKassa-Request-Receipts-ReceiptsRequest.md#method_setLimit) |  | Устанавливает ограничение количества объектов платежа. |
| public | [setPaymentId()](../classes/YooKassa-Request-Receipts-ReceiptsRequest.md#method_setPaymentId) |  | Устанавливает идентификатор платежа или null, если требуется его удалить. |
| public | [setRefundId()](../classes/YooKassa-Request-Receipts-ReceiptsRequest.md#method_setRefundId) |  | Устанавливает идентификатор возврата. |
| public | [setStatus()](../classes/YooKassa-Request-Receipts-ReceiptsRequest.md#method_setStatus) |  | Устанавливает статус выбираемых чеков |
| public | [toArray()](../classes/YooKassa-Common-AbstractObject.md#method_toArray) |  | Возвращает ассоциативный массив со свойствами текущего объекта для его дальнейшей JSON сериализации Является алиасом метода AbstractObject::jsonSerialize(). |
| public | [validate()](../classes/YooKassa-Request-Receipts-ReceiptsRequest.md#method_validate) |  | Проверяет валидность текущего объекта запроса. |
| protected | [getUnknownProperties()](../classes/YooKassa-Common-AbstractObject.md#method_getUnknownProperties) |  | Возвращает массив свойств которые не существуют, но были заданы у объекта. |
| protected | [setValidationError()](../classes/YooKassa-Common-AbstractRequest.md#method_setValidationError) |  | Устанавливает ошибку валидации. |
| protected | [validatePropertyValue()](../classes/YooKassa-Common-AbstractObject.md#method_validatePropertyValue) |  |  |

---
### Details
* File: [lib/Request/Receipts/ReceiptsRequest.php](../../lib/Request/Receipts/ReceiptsRequest.php)
* Package: YooKassa\Request
* Class Hierarchy:  
  * [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)
  * [\YooKassa\Common\AbstractRequest](../classes/YooKassa-Common-AbstractRequest.md)
  * \YooKassa\Request\Receipts\ReceiptsRequest
* Implements:
  * [\YooKassa\Request\Receipts\ReceiptsRequestInterface](../classes/YooKassa-Request-Receipts-ReceiptsRequestInterface.md)

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
<a name="constant_MAX_LIMIT_VALUE" class="anchor"></a>
###### MAX_LIMIT_VALUE
Максимальное количество объектов чеков в выборке

```php
MAX_LIMIT_VALUE = 100
```


<a name="constant_LENGTH_PAYMENT_ID" class="anchor"></a>
###### LENGTH_PAYMENT_ID
Длина идентификатора платежа

```php
LENGTH_PAYMENT_ID = 36
```


<a name="constant_LENGTH_REFUND_ID" class="anchor"></a>
###### LENGTH_REFUND_ID
Длина идентификатора платежа

```php
LENGTH_REFUND_ID = 36
```



---
## Properties
<a name="property_created_at_gt"></a>
#### public $created_at_gt : \DateTime
---
***Description***

Время создания, от (не включая)

**Type:** \DateTime

**Details:**


<a name="property_created_at_gte"></a>
#### public $created_at_gte : \DateTime
---
***Description***

Время создания, от (включительно)

**Type:** \DateTime

**Details:**


<a name="property_created_at_lt"></a>
#### public $created_at_lt : \DateTime
---
***Description***

Время создания, до (не включая)

**Type:** \DateTime

**Details:**


<a name="property_created_at_lte"></a>
#### public $created_at_lte : \DateTime
---
***Description***

Время создания, до (включительно)

**Type:** \DateTime

**Details:**


<a name="property_createdAtGt"></a>
#### public $createdAtGt : \DateTime
---
***Description***

Время создания, от (не включая)

**Type:** \DateTime

**Details:**


<a name="property_createdAtGte"></a>
#### public $createdAtGte : \DateTime
---
***Description***

Время создания, от (включительно)

**Type:** \DateTime

**Details:**


<a name="property_createdAtLt"></a>
#### public $createdAtLt : \DateTime
---
***Description***

Время создания, до (не включая)

**Type:** \DateTime

**Details:**


<a name="property_createdAtLte"></a>
#### public $createdAtLte : \DateTime
---
***Description***

Время создания, до (включительно)

**Type:** \DateTime

**Details:**


<a name="property_cursor"></a>
#### public $cursor : string
---
***Description***

Токен для получения следующей страницы выборки

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**


<a name="property_limit"></a>
#### public $limit : null|int
---
***Description***

Ограничение количества объектов возврата, отображаемых на одной странице выдачи

**Type:** <a href="../null|int"><abbr title="null|int">null|int</abbr></a>

**Details:**


<a name="property_payment_id"></a>
#### public $payment_id : string
---
***Description***

Идентификатор платежа

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**


<a name="property_paymentId"></a>
#### public $paymentId : string
---
***Description***

Идентификатор платежа

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**


<a name="property_refund_id"></a>
#### public $refund_id : string
---
***Description***

Идентификатор возврата

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**


<a name="property_refundId"></a>
#### public $refundId : string
---
***Description***

Идентификатор возврата

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**


<a name="property_status"></a>
#### public $status : string
---
***Description***

Статус возврата

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
#### public builder() : \YooKassa\Request\Receipts\ReceiptsRequestBuilder

```php
Static public builder() : \YooKassa\Request\Receipts\ReceiptsRequestBuilder
```

**Summary**

Возвращает инстанс билдера объектов запросов списка возвратов магазина.

**Details:**
* Inherited From: [\YooKassa\Request\Receipts\ReceiptsRequest](../classes/YooKassa-Request-Receipts-ReceiptsRequest.md)

**Returns:** \YooKassa\Request\Receipts\ReceiptsRequestBuilder - Билдер объектов запросов списка возвратов


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


<a name="method_getCreatedAtGt" class="anchor"></a>
#### public getCreatedAtGt() : null|\DateTime

```php
public getCreatedAtGt() : null|\DateTime
```

**Summary**

Возвращает дату создания от которой будут возвращены возвраты или null, если дата не была установлена.

**Details:**
* Inherited From: [\YooKassa\Request\Receipts\ReceiptsRequest](../classes/YooKassa-Request-Receipts-ReceiptsRequest.md)

**Returns:** null|\DateTime - Время создания, от (не включая)


<a name="method_getCreatedAtGte" class="anchor"></a>
#### public getCreatedAtGte() : null|\DateTime

```php
public getCreatedAtGte() : null|\DateTime
```

**Summary**

Возвращает дату создания от которой будут возвращены возвраты или null, если дата не была установлена.

**Details:**
* Inherited From: [\YooKassa\Request\Receipts\ReceiptsRequest](../classes/YooKassa-Request-Receipts-ReceiptsRequest.md)

**Returns:** null|\DateTime - Время создания, от (включительно)


<a name="method_getCreatedAtLt" class="anchor"></a>
#### public getCreatedAtLt() : null|\DateTime

```php
public getCreatedAtLt() : null|\DateTime
```

**Summary**

Возвращает дату создания до которой будут возвращены возвраты или null, если дата не была установлена.

**Details:**
* Inherited From: [\YooKassa\Request\Receipts\ReceiptsRequest](../classes/YooKassa-Request-Receipts-ReceiptsRequest.md)

**Returns:** null|\DateTime - Время создания, до (не включая)


<a name="method_getCreatedAtLte" class="anchor"></a>
#### public getCreatedAtLte() : null|\DateTime

```php
public getCreatedAtLte() : null|\DateTime
```

**Summary**

Возвращает дату создания до которой будут возвращены возвраты или null, если дата не была установлена.

**Details:**
* Inherited From: [\YooKassa\Request\Receipts\ReceiptsRequest](../classes/YooKassa-Request-Receipts-ReceiptsRequest.md)

**Returns:** null|\DateTime - Время создания, до (включительно)


<a name="method_getCursor" class="anchor"></a>
#### public getCursor() : null|string

```php
public getCursor() : null|string
```

**Summary**

Возвращает токен для получения следующей страницы выборки.

**Details:**
* Inherited From: [\YooKassa\Request\Receipts\ReceiptsRequest](../classes/YooKassa-Request-Receipts-ReceiptsRequest.md)

**Returns:** null|string - Токен для получения следующей страницы выборки


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


<a name="method_getLimit" class="anchor"></a>
#### public getLimit() : null|int

```php
public getLimit() : null|int
```

**Summary**

Ограничение количества объектов платежа.

**Details:**
* Inherited From: [\YooKassa\Request\Receipts\ReceiptsRequest](../classes/YooKassa-Request-Receipts-ReceiptsRequest.md)

**Returns:** null|int - Ограничение количества объектов платежа


<a name="method_getPaymentId" class="anchor"></a>
#### public getPaymentId() : null|string

```php
public getPaymentId() : null|string
```

**Summary**

Возвращает идентификатор платежа если он задан или null.

**Details:**
* Inherited From: [\YooKassa\Request\Receipts\ReceiptsRequest](../classes/YooKassa-Request-Receipts-ReceiptsRequest.md)

**Returns:** null|string - Идентификатор платежа


<a name="method_getRefundId" class="anchor"></a>
#### public getRefundId() : string|null

```php
public getRefundId() : string|null
```

**Summary**

Возвращает идентификатор возврата.

**Details:**
* Inherited From: [\YooKassa\Request\Receipts\ReceiptsRequest](../classes/YooKassa-Request-Receipts-ReceiptsRequest.md)

**Returns:** string|null - Идентификатор возврата


<a name="method_getStatus" class="anchor"></a>
#### public getStatus() : null|string

```php
public getStatus() : null|string
```

**Summary**

Возвращает статус выбираемых возвратов или null, если он до этого не был установлен.

**Details:**
* Inherited From: [\YooKassa\Request\Receipts\ReceiptsRequest](../classes/YooKassa-Request-Receipts-ReceiptsRequest.md)

**Returns:** null|string - Статус выбираемых возвратов


<a name="method_getValidator" class="anchor"></a>
#### public getValidator() : \YooKassa\Validator\Validator

```php
public getValidator() : \YooKassa\Validator\Validator
```

**Details:**
* Inherited From: [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)

**Returns:** \YooKassa\Validator\Validator - 


<a name="method_hasCreatedAtGt" class="anchor"></a>
#### public hasCreatedAtGt() : bool

```php
public hasCreatedAtGt() : bool
```

**Summary**

Проверяет, была ли установлена дата создания от которой выбираются возвраты.

**Details:**
* Inherited From: [\YooKassa\Request\Receipts\ReceiptsRequest](../classes/YooKassa-Request-Receipts-ReceiptsRequest.md)

**Returns:** bool - True если дата была установлена, false если нет


<a name="method_hasCreatedAtGte" class="anchor"></a>
#### public hasCreatedAtGte() : bool

```php
public hasCreatedAtGte() : bool
```

**Summary**

Проверяет, была ли установлена дата создания от которой выбираются возвраты.

**Details:**
* Inherited From: [\YooKassa\Request\Receipts\ReceiptsRequest](../classes/YooKassa-Request-Receipts-ReceiptsRequest.md)

**Returns:** bool - True если дата была установлена, false если нет


<a name="method_hasCreatedAtLt" class="anchor"></a>
#### public hasCreatedAtLt() : bool

```php
public hasCreatedAtLt() : bool
```

**Summary**

Проверяет, была ли установлена дата создания до которой выбираются возвраты.

**Details:**
* Inherited From: [\YooKassa\Request\Receipts\ReceiptsRequest](../classes/YooKassa-Request-Receipts-ReceiptsRequest.md)

**Returns:** bool - True если дата была установлена, false если нет


<a name="method_hasCreatedAtLte" class="anchor"></a>
#### public hasCreatedAtLte() : bool

```php
public hasCreatedAtLte() : bool
```

**Summary**

Проверяет, была ли установлена дата создания до которой выбираются возвраты.

**Details:**
* Inherited From: [\YooKassa\Request\Receipts\ReceiptsRequest](../classes/YooKassa-Request-Receipts-ReceiptsRequest.md)

**Returns:** bool - True если дата была установлена, false если нет


<a name="method_hasCursor" class="anchor"></a>
#### public hasCursor() : bool

```php
public hasCursor() : bool
```

**Summary**

Проверяет, был ли установлен токен следующей страницы.

**Details:**
* Inherited From: [\YooKassa\Request\Receipts\ReceiptsRequest](../classes/YooKassa-Request-Receipts-ReceiptsRequest.md)

**Returns:** bool - True если токен был установлен, false если нет


<a name="method_hasLimit" class="anchor"></a>
#### public hasLimit() : bool

```php
public hasLimit() : bool
```

**Summary**

Проверяет, было ли установлено ограничение количества объектов платежа.

**Details:**
* Inherited From: [\YooKassa\Request\Receipts\ReceiptsRequest](../classes/YooKassa-Request-Receipts-ReceiptsRequest.md)

**Returns:** bool - True если было установлено, false если нет


<a name="method_hasPaymentId" class="anchor"></a>
#### public hasPaymentId() : bool

```php
public hasPaymentId() : bool
```

**Summary**

Проверяет, был ли задан идентификатор платежа.

**Details:**
* Inherited From: [\YooKassa\Request\Receipts\ReceiptsRequest](../classes/YooKassa-Request-Receipts-ReceiptsRequest.md)

**Returns:** bool - True если идентификатор был задан, false если нет


<a name="method_hasRefundId" class="anchor"></a>
#### public hasRefundId() : bool

```php
public hasRefundId() : bool
```

**Summary**

Проверяет, был ли установлен идентификатор возврата.

**Details:**
* Inherited From: [\YooKassa\Request\Receipts\ReceiptsRequest](../classes/YooKassa-Request-Receipts-ReceiptsRequest.md)

**Returns:** bool - True если идентификатор возврата был установлен, false если не был


<a name="method_hasStatus" class="anchor"></a>
#### public hasStatus() : bool

```php
public hasStatus() : bool
```

**Summary**

Проверяет, был ли установлен статус выбираемых возвратов.

**Details:**
* Inherited From: [\YooKassa\Request\Receipts\ReceiptsRequest](../classes/YooKassa-Request-Receipts-ReceiptsRequest.md)

**Returns:** bool - True если статус был установлен, false если нет


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


<a name="method_setCreatedAtGt" class="anchor"></a>
#### public setCreatedAtGt() : self

```php
public setCreatedAtGt(null|\DateTime|int|string $created_at_gt) : self
```

**Summary**

Устанавливает дату создания от которой выбираются возвраты.

**Details:**
* Inherited From: [\YooKassa\Request\Receipts\ReceiptsRequest](../classes/YooKassa-Request-Receipts-ReceiptsRequest.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">null OR \DateTime OR int OR string</code> | created_at_gt  | Время создания, от (не включая) или null, чтобы удалить значение |

**Returns:** self - 


<a name="method_setCreatedAtGte" class="anchor"></a>
#### public setCreatedAtGte() : self

```php
public setCreatedAtGte(null|\DateTime|int|string $_created_at_gte) : self
```

**Summary**

Устанавливает дату создания от которой выбираются возвраты.

**Details:**
* Inherited From: [\YooKassa\Request\Receipts\ReceiptsRequest](../classes/YooKassa-Request-Receipts-ReceiptsRequest.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">null OR \DateTime OR int OR string</code> | _created_at_gte  | Время создания, от (включительно) или null, чтобы удалить значение |

**Returns:** self - 


<a name="method_setCreatedAtLt" class="anchor"></a>
#### public setCreatedAtLt() : self

```php
public setCreatedAtLt(null|\DateTime|int|string $created_at_lt) : self
```

**Summary**

Устанавливает дату создания до которой выбираются возвраты.

**Details:**
* Inherited From: [\YooKassa\Request\Receipts\ReceiptsRequest](../classes/YooKassa-Request-Receipts-ReceiptsRequest.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">null OR \DateTime OR int OR string</code> | created_at_lt  | Время создания, до (не включая) или null, чтобы удалить значение |

**Returns:** self - 


<a name="method_setCreatedAtLte" class="anchor"></a>
#### public setCreatedAtLte() : self

```php
public setCreatedAtLte(null|\DateTime|int|string $created_at_lte) : self
```

**Summary**

Устанавливает дату создания до которой выбираются возвраты.

**Details:**
* Inherited From: [\YooKassa\Request\Receipts\ReceiptsRequest](../classes/YooKassa-Request-Receipts-ReceiptsRequest.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">null OR \DateTime OR int OR string</code> | created_at_lte  | Время создания, до (включительно) или null, чтобы удалить значение |

**Returns:** self - 


<a name="method_setCursor" class="anchor"></a>
#### public setCursor() : self

```php
public setCursor(string|null $cursor) : self
```

**Summary**

Устанавливает токен следующей страницы выборки.

**Details:**
* Inherited From: [\YooKassa\Request\Receipts\ReceiptsRequest](../classes/YooKassa-Request-Receipts-ReceiptsRequest.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | cursor  | Токен следующей страницы выборки или null, чтобы удалить значение |

**Returns:** self - 


<a name="method_setLimit" class="anchor"></a>
#### public setLimit() : self

```php
public setLimit(null|int $limit) : self
```

**Summary**

Устанавливает ограничение количества объектов платежа.

**Details:**
* Inherited From: [\YooKassa\Request\Receipts\ReceiptsRequest](../classes/YooKassa-Request-Receipts-ReceiptsRequest.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">null OR int</code> | limit  | Ограничение количества объектов платежа или null, чтобы удалить значение |

**Returns:** self - 


<a name="method_setPaymentId" class="anchor"></a>
#### public setPaymentId() : self

```php
public setPaymentId(null|string $payment_id) : self
```

**Summary**

Устанавливает идентификатор платежа или null, если требуется его удалить.

**Details:**
* Inherited From: [\YooKassa\Request\Receipts\ReceiptsRequest](../classes/YooKassa-Request-Receipts-ReceiptsRequest.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">null OR string</code> | payment_id  | Идентификатор платежа |

**Returns:** self - 


<a name="method_setRefundId" class="anchor"></a>
#### public setRefundId() : self

```php
public setRefundId(string|null $refund_id) : self
```

**Summary**

Устанавливает идентификатор возврата.

**Details:**
* Inherited From: [\YooKassa\Request\Receipts\ReceiptsRequest](../classes/YooKassa-Request-Receipts-ReceiptsRequest.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | refund_id  | Идентификатор возврата, который ищется в API |

**Returns:** self - 


<a name="method_setStatus" class="anchor"></a>
#### public setStatus() : self

```php
public setStatus(string|null $status) : self
```

**Summary**

Устанавливает статус выбираемых чеков

**Details:**
* Inherited From: [\YooKassa\Request\Receipts\ReceiptsRequest](../classes/YooKassa-Request-Receipts-ReceiptsRequest.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | status  | Статус выбираемых чеков или null, чтобы удалить значение |

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

Проверяет валидность текущего объекта запроса.

**Details:**
* Inherited From: [\YooKassa\Request\Receipts\ReceiptsRequest](../classes/YooKassa-Request-Receipts-ReceiptsRequest.md)

**Returns:** bool - True если объект валиден, false если нет


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