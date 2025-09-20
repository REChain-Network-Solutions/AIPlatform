<?php

/*
 * The MIT License
 *
 * Copyright (c) 2025 "YooMoney", NBСO LLC
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

namespace YooKassa\Model\PersonalData;

use DateTime;
use YooKassa\Common\AbstractObject;
use YooKassa\Model\Metadata;
use YooKassa\Validator\Constraints as Assert;

/**
 * Класс, представляющий модель PersonalData.
 *
 * Объект персональных данных (PersonalData) — актуальная информация о персональных данных пользователя, сохраненных в ЮKassa.
 *
 * @category Class
 * @package  YooKassa\Model
 * @author   cms@yoomoney.ru
 * @link     https://yookassa.ru/developers/api
 * @property string $id Идентификатор персональных данных
 * @property string $type Тип персональных данных
 * @property string $status Текущий статус персональных данных
 * @property DateTime $createdAt Время создания персональных данных
 * @property DateTime $created_at Время создания персональных данных
 * @property null|DateTime $expiresAt Срок жизни объекта персональных данных
 * @property null|DateTime $expires_at Срок жизни объекта персональных данных
 * @property PersonalDataCancellationDetails $cancellationDetails Комментарий к отмене выплаты
 * @property PersonalDataCancellationDetails $cancellation_details Комментарий к отмене выплаты
 * @property Metadata $metadata Метаданные выплаты указанные мерчантом
 */
class PersonalData extends AbstractObject implements PersonalDataInterface
{
    /**
     * Идентификатор персональных данных, сохраненных в ЮKassa.
     *
     * @var string|null
     */
    #[Assert\NotBlank]
    #[Assert\Type('string')]
    #[Assert\Length(max: self::MAX_LENGTH_ID)]
    #[Assert\Length(min: self::MIN_LENGTH_ID)]
    protected ?string $_id = null;

    /**
     * Тип персональных данных — цель, для которой вы будете использовать данные.
     * Возможные значение:
     * - `sbp_payout_recipient` — [выплаты с проверкой получателя](https://yookassa.ru/developers/payouts/scenario-extensions/recipient-check)(только для выплат через СБП);
     * - `payout_statement_recipient` — [выплаты с передачей данных получателя выплаты для выписок из реестра](https://yookassa.ru/developers/payouts/scenario-extensions/recipient-data-send).
     *
     * @var string|null
     */
    #[Assert\NotBlank]
    #[Assert\Choice(callback: [PersonalDataType::class, 'getValidValues'])]
    #[Assert\Type('string')]
    protected ?string $_type = null;

    /**
     * Статус персональных данных.
     * Возможные значения:
     *  * `waiting_for_operation` — данные сохранены, но не использованы при проведении выплаты;
     *  * `active` — данные сохранены и использованы при проведении выплаты; данные можно использовать повторно до срока, указанного в параметре `expires_at`;
     *  * `canceled` — хранение данных отменено, данные удалены, инициатор и причина отмены указаны в объекте `cancellation_details` (финальный и неизменяемый статус).
     *
     * Жизненный цикл персональных данных зависит от назначения данных: [передача данных получателя выплаты](https://yookassa.ru/developers/payouts/scenario-extensions/recipient-data-send#lifecircle)
     * для выписки из реестра или [проверка получателя](https://yookassa.ru/developers/payouts/scenario-extensions/recipient-check#lifecircle) при выплатах через СБП.
     *
     * @var string|null
     */
    #[Assert\NotBlank]
    #[Assert\Choice(callback: [PersonalDataStatus::class, 'getValidValues'])]
    #[Assert\Type('string')]
    protected ?string $_status = null;

    /**
     * Комментарий к статусу canceled: кто и по какой причине аннулировал хранение данных.
     *
     * @var PersonalDataCancellationDetails|null
     */
    #[Assert\Valid]
    #[Assert\Type(PersonalDataCancellationDetails::class)]
    protected ?PersonalDataCancellationDetails $_cancellation_details = null;

    /**
     * Время создания персональных данных.
     * Указывается по [UTC](https://ru.wikipedia.org/wiki/Всемирное_координированное_время) и передается в формате [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601).
     * Пример: ~`2017-11-03T11:52:31.827Z`
     *
     * @var DateTime|null
     */
    #[Assert\NotBlank]
    #[Assert\DateTime(format: YOOKASSA_DATE)]
    #[Assert\Type('DateTime')]
    protected ?DateTime $_created_at = null;

    /**
    Срок жизни объекта персональных данных — время, до которого вы можете использовать персональные данные при проведении операций.
     * Указывается только для объекта в статусе ~`active`. Указывается по [UTC](https://ru.wikipedia.org/wiki/Всемирное_координированное_время)
     * и передается в формате [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601).
     * Пример: ~`2017-11-03T11:52:31.827Z`
     *
     * @var DateTime|null
     */
    #[Assert\DateTime(format: YOOKASSA_DATE)]
    #[Assert\Type('DateTime')]
    protected ?DateTime $_expires_at = null;

    /**
     * Любые дополнительные данные, которые нужны вам для работы (например, ваш внутренний идентификатор заказа).
     *
     * Передаются в виде набора пар «ключ-значение» и возвращаются в ответе от ЮKassa.
     * Ограничения: максимум 16 ключей, имя ключа не больше 32 символов, значение ключа не больше 512 символов, тип данных — строка в формате UTF-8.
     *
     * @var Metadata|null
     */
    #[Assert\Type(Metadata::class)]
    protected ?Metadata $_metadata = null;

    /**
     * Возвращает идентификатор персональных данных, сохраненных в ЮKassa.
     *
     * @return string|null
     */
    public function getId(): ?string
    {
        return $this->_id;
    }

    /**
     * Устанавливает идентификатор персональных данных, сохраненных в ЮKassa.
     *
     * @param string|null $id Идентификатор персональных данных, сохраненных в ЮKassa.
     *
     * @return self
     */
    public function setId(?string $id = null): self
    {
        $this->_id = $this->validatePropertyValue('_id', $id);
        return $this;
    }

    /**
     * Возвращает тип персональных данных.
     *
     * @return string|null Тип персональных данных
     */
    public function getType(): ?string
    {
        return $this->_type;
    }

    /**
     * Устанавливает тип персональных данных.
     *
     * @param string|null $status Тип персональных данных
     *
     * @return self
     */
    public function setType(?string $type = null): self
    {
        $this->_type = $this->validatePropertyValue('_type', $type);
        return $this;
    }

    /**
     * Возвращает статус персональных данных.
     *
     * @return string|null Статус персональных данных
     */
    public function getStatus(): ?string
    {
        return $this->_status;
    }

    /**
     * Устанавливает статус персональных данных.
     *
     * @param string|null $status Статус персональных данных
     *
     * @return $this
     */
    public function setStatus(?string $status = null): self
    {
        $this->_status = $this->validatePropertyValue('_status', $status);
        return $this;
    }

    /**
     * Возвращает комментарий к статусу canceled: кто и по какой причине аннулировал хранение данных.
     *
     * @return PersonalDataCancellationDetails|null Комментарий к статусу canceled
     */
    public function getCancellationDetails(): ?PersonalDataCancellationDetails
    {
        return $this->_cancellation_details;
    }

    /**
     * Устанавливает Комментарий к статусу canceled: кто и по какой причине аннулировал хранение данных.
     *
     * @param PersonalDataCancellationDetails|array|null $cancellation_details Комментарий к статусу canceled
     *
     * @return self
     */
    public function setCancellationDetails(mixed $cancellation_details = null): self
    {
        $this->_cancellation_details = $this->validatePropertyValue('_cancellation_details', $cancellation_details);
        return $this;
    }

    /**
     * Возвращает время создания персональных данных.
     *
     * @return DateTime|null Время создания персональных данных
     */
    public function getCreatedAt(): ?DateTime
    {
        return $this->_created_at;
    }

    /**
     * Устанавливает время создания персональных данных.
     *
     * @param DateTime|string|null $created_at Время создания персональных данных.
     *
     * @return self
     */
    public function setCreatedAt(DateTime|string|null $created_at = null): self
    {
        $this->_created_at = $this->validatePropertyValue('_created_at', $created_at);
        return $this;
    }

    /**
     * Возвращает срок жизни объекта персональных данных.
     *
     * @return DateTime|null Срок жизни объекта персональных данных
     */
    public function getExpiresAt(): ?DateTime
    {
        return $this->_expires_at;
    }

    /**
     * Устанавливает срок жизни объекта персональных данных.
     *
     * @param DateTime|string|null $expires_at Срок жизни объекта персональных данных
     *
     * @return self
     */
    public function setExpiresAt(DateTime|string|null $expires_at = null): self
    {
        $this->_expires_at = $this->validatePropertyValue('_expires_at', $expires_at);
        return $this;
    }

    /**
     * Возвращает любые дополнительные данные.
     *
     * @return Metadata|null Любые дополнительные данные
     */
    public function getMetadata(): ?Metadata
    {
        return $this->_metadata;
    }

    /**
     * Устанавливает любые дополнительные данные.
     *
     * @param Metadata|array|null $metadata Любые дополнительные данные
     *
     * @return self
     */
    public function setMetadata(mixed $metadata = null): self
    {
        $this->_metadata = $this->validatePropertyValue('_metadata', $metadata);
        return $this;
    }
}
