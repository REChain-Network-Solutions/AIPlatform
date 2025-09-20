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
use YooKassa\Model\Metadata;

/**
 * Interface PersonalDataInterface.
 *
 * @category Interface
 * @package  YooKassa\Model
 * @author   cms@yoomoney.ru
 * @link     https://yookassa.ru/developers/api
 *
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
interface PersonalDataInterface
{
    /** @var int Минимальная длина идентификатора */
    public const MIN_LENGTH_ID = 36;

    /** @var int Максимальная длина идентификатора */
    public const MAX_LENGTH_ID = 50;

    /**
     * Возвращает id.
     *
     * @return string|null
     */
    public function getId(): ?string;

    /**
     * Устанавливает id.
     *
     * @param string|null $id идентификатор персональных данных, сохраненных в ЮKassa
     *
     * @return self
     */
    public function setId(?string $id): self;

    /**
     * Возвращает тип персональных данных.
     *
     * @return string|null Тип персональных данных
     */
    public function getType(): ?string;

    /**
     * Устанавливает тип персональных данных.
     *
     * @param string|null $type Тип персональных данных
     *
     * @return self
     */
    public function setType(?string $type): self;

    /**
     * Возвращает статус персональных данных.
     *
     * @return string|null Статус персональных данных
     */
    public function getStatus(): ?string;

    /**
     * Устанавливает статус персональных данных.
     *
     * @param string|null $status Статус персональных данных
     *
     * @return self
     */
    public function setStatus(?string $status): self;

    /**
     * Возвращает cancellation_details.
     */
    public function getCancellationDetails(): ?PersonalDataCancellationDetails;

    /**
     * Устанавливает cancellation_details.
     *
     * @param null|array|PersonalDataCancellationDetails $cancellation_details
     *
     * @return self
     */
    public function setCancellationDetails(mixed $cancellation_details): self;

    /**
     * Возвращает created_at.
     *
     * @return DateTime|null
     */
    public function getCreatedAt(): ?DateTime;

    /**
     * Устанавливает время создания персональных данных.
     *
     * @param DateTime|string|null $created_at Время создания персональных данных
     *
     * @return self
     */
    public function setCreatedAt(DateTime|string|null $created_at): self;

    /**
     * Возвращает expires_at.
     *
     * @return DateTime|null
     */
    public function getExpiresAt(): ?DateTime;

    /**
     * Устанавливает срок жизни объекта персональных данных.
     *
     * @param DateTime|string|null $expires_at Срок жизни объекта персональных данных
     *
     * @return self
     */
    public function setExpiresAt(DateTime|string|null $expires_at = null): self;

    /**
     * Возвращает metadata.
     */
    public function getMetadata(): ?Metadata;

    /**
     * Устанавливает metadata.
     *
     * @param null|array|Metadata $metadata любые дополнительные данные, которые нужны вам для работы (например, ваш внутренний идентификатор заказа)
     *
     * @return self
     */
    public function setMetadata(mixed $metadata = null): self;
}
