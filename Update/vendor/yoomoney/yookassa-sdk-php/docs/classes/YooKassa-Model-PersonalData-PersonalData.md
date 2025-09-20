# [YooKassa API SDK](../home.md)

# Class: \YooKassa\Model\PersonalData\PersonalData
### Namespace: [\YooKassa\Model\PersonalData](../namespaces/yookassa-model-personaldata.md)
---
**Summary:**

Класс, представляющий модель PersonalData.

**Description:**

Объект персональных данных (PersonalData) — актуальная информация о персональных данных пользователя, сохраненных в ЮKassa.

---
### Constants
* No constants found

---
### Properties
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| public | [$cancellation_details](../classes/YooKassa-Model-PersonalData-PersonalData.md#property_cancellation_details) |  | Комментарий к отмене выплаты |
| public | [$cancellationDetails](../classes/YooKassa-Model-PersonalData-PersonalData.md#property_cancellationDetails) |  | Комментарий к отмене выплаты |
| public | [$created_at](../classes/YooKassa-Model-PersonalData-PersonalData.md#property_created_at) |  | Время создания персональных данных |
| public | [$createdAt](../classes/YooKassa-Model-PersonalData-PersonalData.md#property_createdAt) |  | Время создания персональных данных |
| public | [$expires_at](../classes/YooKassa-Model-PersonalData-PersonalData.md#property_expires_at) |  | Срок жизни объекта персональных данных |
| public | [$expiresAt](../classes/YooKassa-Model-PersonalData-PersonalData.md#property_expiresAt) |  | Срок жизни объекта персональных данных |
| public | [$id](../classes/YooKassa-Model-PersonalData-PersonalData.md#property_id) |  | Идентификатор персональных данных |
| public | [$metadata](../classes/YooKassa-Model-PersonalData-PersonalData.md#property_metadata) |  | Метаданные выплаты указанные мерчантом |
| public | [$status](../classes/YooKassa-Model-PersonalData-PersonalData.md#property_status) |  | Текущий статус персональных данных |
| public | [$type](../classes/YooKassa-Model-PersonalData-PersonalData.md#property_type) |  | Тип персональных данных |
| protected | [$_cancellation_details](../classes/YooKassa-Model-PersonalData-PersonalData.md#property__cancellation_details) |  | Комментарий к статусу canceled: кто и по какой причине аннулировал хранение данных. |
| protected | [$_created_at](../classes/YooKassa-Model-PersonalData-PersonalData.md#property__created_at) |  | Время создания персональных данных. |
| protected | [$_expires_at](../classes/YooKassa-Model-PersonalData-PersonalData.md#property__expires_at) |  | Срок жизни объекта персональных данных — время, до которого вы можете использовать персональные данные при проведении операций. |
| protected | [$_id](../classes/YooKassa-Model-PersonalData-PersonalData.md#property__id) |  | Идентификатор персональных данных, сохраненных в ЮKassa. |
| protected | [$_metadata](../classes/YooKassa-Model-PersonalData-PersonalData.md#property__metadata) |  | Любые дополнительные данные, которые нужны вам для работы (например, ваш внутренний идентификатор заказа). |
| protected | [$_status](../classes/YooKassa-Model-PersonalData-PersonalData.md#property__status) |  | Статус персональных данных. |
| protected | [$_type](../classes/YooKassa-Model-PersonalData-PersonalData.md#property__type) |  | Тип персональных данных — цель, для которой вы будете использовать данные. |

---
### Methods
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| public | [__construct()](../classes/YooKassa-Common-AbstractObject.md#method___construct) |  | AbstractObject constructor. |
| public | [__get()](../classes/YooKassa-Common-AbstractObject.md#method___get) |  | Возвращает значение свойства. |
| public | [__isset()](../classes/YooKassa-Common-AbstractObject.md#method___isset) |  | Проверяет наличие свойства. |
| public | [__set()](../classes/YooKassa-Common-AbstractObject.md#method___set) |  | Устанавливает значение свойства. |
| public | [__unset()](../classes/YooKassa-Common-AbstractObject.md#method___unset) |  | Удаляет свойство. |
| public | [fromArray()](../classes/YooKassa-Common-AbstractObject.md#method_fromArray) |  | Устанавливает значения свойств текущего объекта из массива. |
| public | [getCancellationDetails()](../classes/YooKassa-Model-PersonalData-PersonalData.md#method_getCancellationDetails) |  | Возвращает комментарий к статусу canceled: кто и по какой причине аннулировал хранение данных. |
| public | [getCreatedAt()](../classes/YooKassa-Model-PersonalData-PersonalData.md#method_getCreatedAt) |  | Возвращает время создания персональных данных. |
| public | [getExpiresAt()](../classes/YooKassa-Model-PersonalData-PersonalData.md#method_getExpiresAt) |  | Возвращает срок жизни объекта персональных данных. |
| public | [getId()](../classes/YooKassa-Model-PersonalData-PersonalData.md#method_getId) |  | Возвращает идентификатор персональных данных, сохраненных в ЮKassa. |
| public | [getMetadata()](../classes/YooKassa-Model-PersonalData-PersonalData.md#method_getMetadata) |  | Возвращает любые дополнительные данные. |
| public | [getStatus()](../classes/YooKassa-Model-PersonalData-PersonalData.md#method_getStatus) |  | Возвращает статус персональных данных. |
| public | [getType()](../classes/YooKassa-Model-PersonalData-PersonalData.md#method_getType) |  | Возвращает тип персональных данных. |
| public | [getValidator()](../classes/YooKassa-Common-AbstractObject.md#method_getValidator) |  |  |
| public | [jsonSerialize()](../classes/YooKassa-Common-AbstractObject.md#method_jsonSerialize) |  | Возвращает ассоциативный массив со свойствами текущего объекта для его дальнейшей JSON сериализации. |
| public | [offsetExists()](../classes/YooKassa-Common-AbstractObject.md#method_offsetExists) |  | Проверяет наличие свойства. |
| public | [offsetGet()](../classes/YooKassa-Common-AbstractObject.md#method_offsetGet) |  | Возвращает значение свойства. |
| public | [offsetSet()](../classes/YooKassa-Common-AbstractObject.md#method_offsetSet) |  | Устанавливает значение свойства. |
| public | [offsetUnset()](../classes/YooKassa-Common-AbstractObject.md#method_offsetUnset) |  | Удаляет свойство. |
| public | [setCancellationDetails()](../classes/YooKassa-Model-PersonalData-PersonalData.md#method_setCancellationDetails) |  | Устанавливает Комментарий к статусу canceled: кто и по какой причине аннулировал хранение данных. |
| public | [setCreatedAt()](../classes/YooKassa-Model-PersonalData-PersonalData.md#method_setCreatedAt) |  | Устанавливает время создания персональных данных. |
| public | [setExpiresAt()](../classes/YooKassa-Model-PersonalData-PersonalData.md#method_setExpiresAt) |  | Устанавливает срок жизни объекта персональных данных. |
| public | [setId()](../classes/YooKassa-Model-PersonalData-PersonalData.md#method_setId) |  | Устанавливает идентификатор персональных данных, сохраненных в ЮKassa. |
| public | [setMetadata()](../classes/YooKassa-Model-PersonalData-PersonalData.md#method_setMetadata) |  | Устанавливает любые дополнительные данные. |
| public | [setStatus()](../classes/YooKassa-Model-PersonalData-PersonalData.md#method_setStatus) |  | Устанавливает статус персональных данных. |
| public | [setType()](../classes/YooKassa-Model-PersonalData-PersonalData.md#method_setType) |  | Устанавливает тип персональных данных. |
| public | [toArray()](../classes/YooKassa-Common-AbstractObject.md#method_toArray) |  | Возвращает ассоциативный массив со свойствами текущего объекта для его дальнейшей JSON сериализации Является алиасом метода AbstractObject::jsonSerialize(). |
| protected | [getUnknownProperties()](../classes/YooKassa-Common-AbstractObject.md#method_getUnknownProperties) |  | Возвращает массив свойств которые не существуют, но были заданы у объекта. |
| protected | [validatePropertyValue()](../classes/YooKassa-Common-AbstractObject.md#method_validatePropertyValue) |  |  |

---
### Details
* File: [lib/Model/PersonalData/PersonalData.php](../../lib/Model/PersonalData/PersonalData.php)
* Package: YooKassa\Model
* Class Hierarchy: 
  * [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)
  * \YooKassa\Model\PersonalData\PersonalData
* Implements:
  * [\YooKassa\Model\PersonalData\PersonalDataInterface](../classes/YooKassa-Model-PersonalData-PersonalDataInterface.md)

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
<a name="property_cancellation_details"></a>
#### public $cancellation_details : \YooKassa\Model\PersonalData\PersonalDataCancellationDetails
---
***Description***

Комментарий к отмене выплаты

**Type:** <a href="../classes/YooKassa-Model-PersonalData-PersonalDataCancellationDetails.html"><abbr title="\YooKassa\Model\PersonalData\PersonalDataCancellationDetails">PersonalDataCancellationDetails</abbr></a>

**Details:**


<a name="property_cancellationDetails"></a>
#### public $cancellationDetails : \YooKassa\Model\PersonalData\PersonalDataCancellationDetails
---
***Description***

Комментарий к отмене выплаты

**Type:** <a href="../classes/YooKassa-Model-PersonalData-PersonalDataCancellationDetails.html"><abbr title="\YooKassa\Model\PersonalData\PersonalDataCancellationDetails">PersonalDataCancellationDetails</abbr></a>

**Details:**


<a name="property_created_at"></a>
#### public $created_at : \DateTime
---
***Description***

Время создания персональных данных

**Type:** \DateTime

**Details:**


<a name="property_createdAt"></a>
#### public $createdAt : \DateTime
---
***Description***

Время создания персональных данных

**Type:** \DateTime

**Details:**


<a name="property_expires_at"></a>
#### public $expires_at : null|\DateTime
---
***Description***

Срок жизни объекта персональных данных

**Type:** <a href="../null|\DateTime"><abbr title="null|\DateTime">DateTime</abbr></a>

**Details:**


<a name="property_expiresAt"></a>
#### public $expiresAt : null|\DateTime
---
***Description***

Срок жизни объекта персональных данных

**Type:** <a href="../null|\DateTime"><abbr title="null|\DateTime">DateTime</abbr></a>

**Details:**


<a name="property_id"></a>
#### public $id : string
---
***Description***

Идентификатор персональных данных

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**


<a name="property_metadata"></a>
#### public $metadata : \YooKassa\Model\Metadata
---
***Description***

Метаданные выплаты указанные мерчантом

**Type:** <a href="../classes/YooKassa-Model-Metadata.html"><abbr title="\YooKassa\Model\Metadata">Metadata</abbr></a>

**Details:**


<a name="property_status"></a>
#### public $status : string
---
***Description***

Текущий статус персональных данных

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**


<a name="property_type"></a>
#### public $type : string
---
***Description***

Тип персональных данных

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**


<a name="property__cancellation_details"></a>
#### protected $_cancellation_details : ?\YooKassa\Model\PersonalData\PersonalDataCancellationDetails
---
**Summary**

Комментарий к статусу canceled: кто и по какой причине аннулировал хранение данных.

**Type:** <a href="../?\YooKassa\Model\PersonalData\PersonalDataCancellationDetails"><abbr title="?\YooKassa\Model\PersonalData\PersonalDataCancellationDetails">PersonalDataCancellationDetails</abbr></a>

**Details:**


<a name="property__created_at"></a>
#### protected $_created_at : ?\DateTime
---
**Summary**

Время создания персональных данных.

***Description***

Указывается по [UTC](https://ru.wikipedia.org/wiki/Всемирное_координированное_время) и передается в формате [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601).
Пример: ~`2017-11-03T11:52:31.827Z`

**Type:** <a href="../?\DateTime"><abbr title="?\DateTime">DateTime</abbr></a>

**Details:**


<a name="property__expires_at"></a>
#### protected $_expires_at : ?\DateTime
---
**Summary**

Срок жизни объекта персональных данных — время, до которого вы можете использовать персональные данные при проведении операций.

***Description***

Указывается только для объекта в статусе ~`active`. Указывается по [UTC](https://ru.wikipedia.org/wiki/Всемирное_координированное_время)
и передается в формате [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601).
Пример: ~`2017-11-03T11:52:31.827Z`

**Type:** <a href="../?\DateTime"><abbr title="?\DateTime">DateTime</abbr></a>

**Details:**


<a name="property__id"></a>
#### protected $_id : ?string
---
**Summary**

Идентификатор персональных данных, сохраненных в ЮKassa.

**Type:** <a href="../?string"><abbr title="?string">?string</abbr></a>

**Details:**


<a name="property__metadata"></a>
#### protected $_metadata : ?\YooKassa\Model\Metadata
---
**Summary**

Любые дополнительные данные, которые нужны вам для работы (например, ваш внутренний идентификатор заказа).

***Description***

Передаются в виде набора пар «ключ-значение» и возвращаются в ответе от ЮKassa.
Ограничения: максимум 16 ключей, имя ключа не больше 32 символов, значение ключа не больше 512 символов, тип данных — строка в формате UTF-8.

**Type:** <a href="../?\YooKassa\Model\Metadata"><abbr title="?\YooKassa\Model\Metadata">Metadata</abbr></a>

**Details:**


<a name="property__status"></a>
#### protected $_status : ?string
---
**Summary**

Статус персональных данных.

***Description***

Возможные значения:
 * `waiting_for_operation` — данные сохранены, но не использованы при проведении выплаты;
 * `active` — данные сохранены и использованы при проведении выплаты; данные можно использовать повторно до срока, указанного в параметре `expires_at`;
 * `canceled` — хранение данных отменено, данные удалены, инициатор и причина отмены указаны в объекте `cancellation_details` (финальный и неизменяемый статус).

Жизненный цикл персональных данных зависит от назначения данных: [передача данных получателя выплаты](https://yookassa.ru/developers/payouts/scenario-extensions/recipient-data-send#lifecircle)
для выписки из реестра или [проверка получателя](https://yookassa.ru/developers/payouts/scenario-extensions/recipient-check#lifecircle) при выплатах через СБП.

**Type:** <a href="../?string"><abbr title="?string">?string</abbr></a>

**Details:**


<a name="property__type"></a>
#### protected $_type : ?string
---
**Summary**

Тип персональных данных — цель, для которой вы будете использовать данные.

***Description***

Возможные значение:
- `sbp_payout_recipient` — [выплаты с проверкой получателя](https://yookassa.ru/developers/payouts/scenario-extensions/recipient-check)(только для выплат через СБП);
- `payout_statement_recipient` — [выплаты с передачей данных получателя выплаты для выписок из реестра](https://yookassa.ru/developers/payouts/scenario-extensions/recipient-data-send).

**Type:** <a href="../?string"><abbr title="?string">?string</abbr></a>

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


<a name="method_getCancellationDetails" class="anchor"></a>
#### public getCancellationDetails() : \YooKassa\Model\PersonalData\PersonalDataCancellationDetails|null

```php
public getCancellationDetails() : \YooKassa\Model\PersonalData\PersonalDataCancellationDetails|null
```

**Summary**

Возвращает комментарий к статусу canceled: кто и по какой причине аннулировал хранение данных.

**Details:**
* Inherited From: [\YooKassa\Model\PersonalData\PersonalData](../classes/YooKassa-Model-PersonalData-PersonalData.md)

**Returns:** \YooKassa\Model\PersonalData\PersonalDataCancellationDetails|null - Комментарий к статусу canceled


<a name="method_getCreatedAt" class="anchor"></a>
#### public getCreatedAt() : \DateTime|null

```php
public getCreatedAt() : \DateTime|null
```

**Summary**

Возвращает время создания персональных данных.

**Details:**
* Inherited From: [\YooKassa\Model\PersonalData\PersonalData](../classes/YooKassa-Model-PersonalData-PersonalData.md)

**Returns:** \DateTime|null - Время создания персональных данных


<a name="method_getExpiresAt" class="anchor"></a>
#### public getExpiresAt() : \DateTime|null

```php
public getExpiresAt() : \DateTime|null
```

**Summary**

Возвращает срок жизни объекта персональных данных.

**Details:**
* Inherited From: [\YooKassa\Model\PersonalData\PersonalData](../classes/YooKassa-Model-PersonalData-PersonalData.md)

**Returns:** \DateTime|null - Срок жизни объекта персональных данных


<a name="method_getId" class="anchor"></a>
#### public getId() : string|null

```php
public getId() : string|null
```

**Summary**

Возвращает идентификатор персональных данных, сохраненных в ЮKassa.

**Details:**
* Inherited From: [\YooKassa\Model\PersonalData\PersonalData](../classes/YooKassa-Model-PersonalData-PersonalData.md)

**Returns:** string|null - 


<a name="method_getMetadata" class="anchor"></a>
#### public getMetadata() : \YooKassa\Model\Metadata|null

```php
public getMetadata() : \YooKassa\Model\Metadata|null
```

**Summary**

Возвращает любые дополнительные данные.

**Details:**
* Inherited From: [\YooKassa\Model\PersonalData\PersonalData](../classes/YooKassa-Model-PersonalData-PersonalData.md)

**Returns:** \YooKassa\Model\Metadata|null - Любые дополнительные данные


<a name="method_getStatus" class="anchor"></a>
#### public getStatus() : string|null

```php
public getStatus() : string|null
```

**Summary**

Возвращает статус персональных данных.

**Details:**
* Inherited From: [\YooKassa\Model\PersonalData\PersonalData](../classes/YooKassa-Model-PersonalData-PersonalData.md)

**Returns:** string|null - Статус персональных данных


<a name="method_getType" class="anchor"></a>
#### public getType() : string|null

```php
public getType() : string|null
```

**Summary**

Возвращает тип персональных данных.

**Details:**
* Inherited From: [\YooKassa\Model\PersonalData\PersonalData](../classes/YooKassa-Model-PersonalData-PersonalData.md)

**Returns:** string|null - Тип персональных данных


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


<a name="method_setCancellationDetails" class="anchor"></a>
#### public setCancellationDetails() : self

```php
public setCancellationDetails(\YooKassa\Model\PersonalData\PersonalDataCancellationDetails|array|null $cancellation_details = null) : self
```

**Summary**

Устанавливает Комментарий к статусу canceled: кто и по какой причине аннулировал хранение данных.

**Details:**
* Inherited From: [\YooKassa\Model\PersonalData\PersonalData](../classes/YooKassa-Model-PersonalData-PersonalData.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">\YooKassa\Model\PersonalData\PersonalDataCancellationDetails OR array OR null</code> | cancellation_details  | Комментарий к статусу canceled |

**Returns:** self - 


<a name="method_setCreatedAt" class="anchor"></a>
#### public setCreatedAt() : self

```php
public setCreatedAt(\DateTime|string|null $created_at = null) : self
```

**Summary**

Устанавливает время создания персональных данных.

**Details:**
* Inherited From: [\YooKassa\Model\PersonalData\PersonalData](../classes/YooKassa-Model-PersonalData-PersonalData.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">\DateTime OR string OR null</code> | created_at  | Время создания персональных данных. |

**Returns:** self - 


<a name="method_setExpiresAt" class="anchor"></a>
#### public setExpiresAt() : self

```php
public setExpiresAt(\DateTime|string|null $expires_at = null) : self
```

**Summary**

Устанавливает срок жизни объекта персональных данных.

**Details:**
* Inherited From: [\YooKassa\Model\PersonalData\PersonalData](../classes/YooKassa-Model-PersonalData-PersonalData.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">\DateTime OR string OR null</code> | expires_at  | Срок жизни объекта персональных данных |

**Returns:** self - 


<a name="method_setId" class="anchor"></a>
#### public setId() : self

```php
public setId(string|null $id = null) : self
```

**Summary**

Устанавливает идентификатор персональных данных, сохраненных в ЮKassa.

**Details:**
* Inherited From: [\YooKassa\Model\PersonalData\PersonalData](../classes/YooKassa-Model-PersonalData-PersonalData.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | id  | Идентификатор персональных данных, сохраненных в ЮKassa. |

**Returns:** self - 


<a name="method_setMetadata" class="anchor"></a>
#### public setMetadata() : self

```php
public setMetadata(\YooKassa\Model\Metadata|array|null $metadata = null) : self
```

**Summary**

Устанавливает любые дополнительные данные.

**Details:**
* Inherited From: [\YooKassa\Model\PersonalData\PersonalData](../classes/YooKassa-Model-PersonalData-PersonalData.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">\YooKassa\Model\Metadata OR array OR null</code> | metadata  | Любые дополнительные данные |

**Returns:** self - 


<a name="method_setStatus" class="anchor"></a>
#### public setStatus() : $this

```php
public setStatus(string|null $status = null) : $this
```

**Summary**

Устанавливает статус персональных данных.

**Details:**
* Inherited From: [\YooKassa\Model\PersonalData\PersonalData](../classes/YooKassa-Model-PersonalData-PersonalData.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | status  | Статус персональных данных |

**Returns:** $this - 


<a name="method_setType" class="anchor"></a>
#### public setType() : self

```php
public setType(?string $type = null) : self
```

**Summary**

Устанавливает тип персональных данных.

**Details:**
* Inherited From: [\YooKassa\Model\PersonalData\PersonalData](../classes/YooKassa-Model-PersonalData-PersonalData.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">?string</code> | type  | Тип персональных данных |

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