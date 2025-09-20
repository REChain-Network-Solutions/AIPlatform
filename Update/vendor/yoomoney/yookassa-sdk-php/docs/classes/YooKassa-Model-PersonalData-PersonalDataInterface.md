# [YooKassa API SDK](../home.md)

# Interface: PersonalDataInterface
### Namespace: [\YooKassa\Model\PersonalData](../namespaces/yookassa-model-personaldata.md)
---
**Summary:**

Interface PersonalDataInterface.

---
### Constants
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
* [public MIN_LENGTH_ID](../classes/YooKassa-Model-PersonalData-PersonalDataInterface.md#constant_MIN_LENGTH_ID)
* [public MAX_LENGTH_ID](../classes/YooKassa-Model-PersonalData-PersonalDataInterface.md#constant_MAX_LENGTH_ID)

---
### Methods
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| public | [getCancellationDetails()](../classes/YooKassa-Model-PersonalData-PersonalDataInterface.md#method_getCancellationDetails) |  | Возвращает cancellation_details. |
| public | [getCreatedAt()](../classes/YooKassa-Model-PersonalData-PersonalDataInterface.md#method_getCreatedAt) |  | Возвращает created_at. |
| public | [getExpiresAt()](../classes/YooKassa-Model-PersonalData-PersonalDataInterface.md#method_getExpiresAt) |  | Возвращает expires_at. |
| public | [getId()](../classes/YooKassa-Model-PersonalData-PersonalDataInterface.md#method_getId) |  | Возвращает id. |
| public | [getMetadata()](../classes/YooKassa-Model-PersonalData-PersonalDataInterface.md#method_getMetadata) |  | Возвращает metadata. |
| public | [getStatus()](../classes/YooKassa-Model-PersonalData-PersonalDataInterface.md#method_getStatus) |  | Возвращает статус персональных данных. |
| public | [getType()](../classes/YooKassa-Model-PersonalData-PersonalDataInterface.md#method_getType) |  | Возвращает тип персональных данных. |
| public | [setCancellationDetails()](../classes/YooKassa-Model-PersonalData-PersonalDataInterface.md#method_setCancellationDetails) |  | Устанавливает cancellation_details. |
| public | [setCreatedAt()](../classes/YooKassa-Model-PersonalData-PersonalDataInterface.md#method_setCreatedAt) |  | Устанавливает время создания персональных данных. |
| public | [setExpiresAt()](../classes/YooKassa-Model-PersonalData-PersonalDataInterface.md#method_setExpiresAt) |  | Устанавливает срок жизни объекта персональных данных. |
| public | [setId()](../classes/YooKassa-Model-PersonalData-PersonalDataInterface.md#method_setId) |  | Устанавливает id. |
| public | [setMetadata()](../classes/YooKassa-Model-PersonalData-PersonalDataInterface.md#method_setMetadata) |  | Устанавливает metadata. |
| public | [setStatus()](../classes/YooKassa-Model-PersonalData-PersonalDataInterface.md#method_setStatus) |  | Устанавливает статус персональных данных. |
| public | [setType()](../classes/YooKassa-Model-PersonalData-PersonalDataInterface.md#method_setType) |  | Устанавливает тип персональных данных. |

---
### Details
* File: [lib/Model/PersonalData/PersonalDataInterface.php](../../lib/Model/PersonalData/PersonalDataInterface.php)
* Package: \YooKassa\Model
* See Also:
  * [](https://yookassa.ru/developers/api)

---
### Tags
| Tag | Version | Description |
| --- | ------- | ----------- |
| category |  | Interface |
| author |  | cms@yoomoney.ru |
| property |  | Идентификатор персональных данных |
| property |  | Тип персональных данных |
| property |  | Текущий статус персональных данных |
| property |  | Время создания персональных данных |
| property |  | Время создания персональных данных |
| property |  | Срок жизни объекта персональных данных |
| property |  | Срок жизни объекта персональных данных |
| property |  | Комментарий к отмене выплаты |
| property |  | Комментарий к отмене выплаты |
| property |  | Метаданные выплаты указанные мерчантом |

---
## Constants
<a name="constant_MIN_LENGTH_ID" class="anchor"></a>
###### MIN_LENGTH_ID
```php
MIN_LENGTH_ID = 36 : int
```


<a name="constant_MAX_LENGTH_ID" class="anchor"></a>
###### MAX_LENGTH_ID
```php
MAX_LENGTH_ID = 50 : int
```



---
## Methods
<a name="method_getId" class="anchor"></a>
#### public getId() : string|null

```php
public getId() : string|null
```

**Summary**

Возвращает id.

**Details:**
* Inherited From: [\YooKassa\Model\PersonalData\PersonalDataInterface](../classes/YooKassa-Model-PersonalData-PersonalDataInterface.md)

**Returns:** string|null - 


<a name="method_setId" class="anchor"></a>
#### public setId() : self

```php
public setId(string|null $id) : self
```

**Summary**

Устанавливает id.

**Details:**
* Inherited From: [\YooKassa\Model\PersonalData\PersonalDataInterface](../classes/YooKassa-Model-PersonalData-PersonalDataInterface.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | id  | идентификатор персональных данных, сохраненных в ЮKassa |

**Returns:** self - 


<a name="method_getType" class="anchor"></a>
#### public getType() : string|null

```php
public getType() : string|null
```

**Summary**

Возвращает тип персональных данных.

**Details:**
* Inherited From: [\YooKassa\Model\PersonalData\PersonalDataInterface](../classes/YooKassa-Model-PersonalData-PersonalDataInterface.md)

**Returns:** string|null - Тип персональных данных


<a name="method_setType" class="anchor"></a>
#### public setType() : self

```php
public setType(string|null $type) : self
```

**Summary**

Устанавливает тип персональных данных.

**Details:**
* Inherited From: [\YooKassa\Model\PersonalData\PersonalDataInterface](../classes/YooKassa-Model-PersonalData-PersonalDataInterface.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | type  | Тип персональных данных |

**Returns:** self - 


<a name="method_getStatus" class="anchor"></a>
#### public getStatus() : string|null

```php
public getStatus() : string|null
```

**Summary**

Возвращает статус персональных данных.

**Details:**
* Inherited From: [\YooKassa\Model\PersonalData\PersonalDataInterface](../classes/YooKassa-Model-PersonalData-PersonalDataInterface.md)

**Returns:** string|null - Статус персональных данных


<a name="method_setStatus" class="anchor"></a>
#### public setStatus() : self

```php
public setStatus(string|null $status) : self
```

**Summary**

Устанавливает статус персональных данных.

**Details:**
* Inherited From: [\YooKassa\Model\PersonalData\PersonalDataInterface](../classes/YooKassa-Model-PersonalData-PersonalDataInterface.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | status  | Статус персональных данных |

**Returns:** self - 


<a name="method_getCancellationDetails" class="anchor"></a>
#### public getCancellationDetails() : ?\YooKassa\Model\PersonalData\PersonalDataCancellationDetails

```php
public getCancellationDetails() : ?\YooKassa\Model\PersonalData\PersonalDataCancellationDetails
```

**Summary**

Возвращает cancellation_details.

**Details:**
* Inherited From: [\YooKassa\Model\PersonalData\PersonalDataInterface](../classes/YooKassa-Model-PersonalData-PersonalDataInterface.md)

**Returns:** ?\YooKassa\Model\PersonalData\PersonalDataCancellationDetails - 


<a name="method_setCancellationDetails" class="anchor"></a>
#### public setCancellationDetails() : self

```php
public setCancellationDetails(null|array|\YooKassa\Model\PersonalData\PersonalDataCancellationDetails $cancellation_details) : self
```

**Summary**

Устанавливает cancellation_details.

**Details:**
* Inherited From: [\YooKassa\Model\PersonalData\PersonalDataInterface](../classes/YooKassa-Model-PersonalData-PersonalDataInterface.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">null OR array OR \YooKassa\Model\PersonalData\PersonalDataCancellationDetails</code> | cancellation_details  |  |

**Returns:** self - 


<a name="method_getCreatedAt" class="anchor"></a>
#### public getCreatedAt() : \DateTime|null

```php
public getCreatedAt() : \DateTime|null
```

**Summary**

Возвращает created_at.

**Details:**
* Inherited From: [\YooKassa\Model\PersonalData\PersonalDataInterface](../classes/YooKassa-Model-PersonalData-PersonalDataInterface.md)

**Returns:** \DateTime|null - 


<a name="method_setCreatedAt" class="anchor"></a>
#### public setCreatedAt() : self

```php
public setCreatedAt(\DateTime|string|null $created_at) : self
```

**Summary**

Устанавливает время создания персональных данных.

**Details:**
* Inherited From: [\YooKassa\Model\PersonalData\PersonalDataInterface](../classes/YooKassa-Model-PersonalData-PersonalDataInterface.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">\DateTime OR string OR null</code> | created_at  | Время создания персональных данных |

**Returns:** self - 


<a name="method_getExpiresAt" class="anchor"></a>
#### public getExpiresAt() : \DateTime|null

```php
public getExpiresAt() : \DateTime|null
```

**Summary**

Возвращает expires_at.

**Details:**
* Inherited From: [\YooKassa\Model\PersonalData\PersonalDataInterface](../classes/YooKassa-Model-PersonalData-PersonalDataInterface.md)

**Returns:** \DateTime|null - 


<a name="method_setExpiresAt" class="anchor"></a>
#### public setExpiresAt() : self

```php
public setExpiresAt(\DateTime|string|null $expires_at = null) : self
```

**Summary**

Устанавливает срок жизни объекта персональных данных.

**Details:**
* Inherited From: [\YooKassa\Model\PersonalData\PersonalDataInterface](../classes/YooKassa-Model-PersonalData-PersonalDataInterface.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">\DateTime OR string OR null</code> | expires_at  | Срок жизни объекта персональных данных |

**Returns:** self - 


<a name="method_getMetadata" class="anchor"></a>
#### public getMetadata() : ?\YooKassa\Model\Metadata

```php
public getMetadata() : ?\YooKassa\Model\Metadata
```

**Summary**

Возвращает metadata.

**Details:**
* Inherited From: [\YooKassa\Model\PersonalData\PersonalDataInterface](../classes/YooKassa-Model-PersonalData-PersonalDataInterface.md)

**Returns:** ?\YooKassa\Model\Metadata - 


<a name="method_setMetadata" class="anchor"></a>
#### public setMetadata() : self

```php
public setMetadata(null|array|\YooKassa\Model\Metadata $metadata = null) : self
```

**Summary**

Устанавливает metadata.

**Details:**
* Inherited From: [\YooKassa\Model\PersonalData\PersonalDataInterface](../classes/YooKassa-Model-PersonalData-PersonalDataInterface.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">null OR array OR \YooKassa\Model\Metadata</code> | metadata  | любые дополнительные данные, которые нужны вам для работы (например, ваш внутренний идентификатор заказа) |

**Returns:** self - 




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