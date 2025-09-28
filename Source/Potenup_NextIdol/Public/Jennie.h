#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Character.h"
#include "Jennie.generated.h"

class UAnimSequence;

UCLASS()
class POTENUP_NEXTIDOL_API AJennie : public ACharacter
{
    GENERATED_BODY()

public:
    AJennie();

    UFUNCTION(BlueprintCallable, Category="Performance")
    void StartSongPerformance();

    UFUNCTION(BlueprintCallable, Category="Performance")
    void NotifySongStarted(int32 SongIndex);

protected:
    virtual void BeginPlay() override;

    UPROPERTY(EditAnywhere, Category="Performance", meta=(AllowPrivateAccess="true"))
    FVector SongStartLocation = FVector(20.f, 510.f, 300.f);

    UPROPERTY(EditAnywhere, Category="Performance", meta=(AllowPrivateAccess="true"))
    int32 TargetSongIndex = 2;

    UPROPERTY(EditAnywhere, Category="Performance", meta=(AllowPrivateAccess="true"))
    bool bAutoStartOnBeginPlay = false;

    UPROPERTY(EditAnywhere, Category="Performance", meta=(AllowPrivateAccess="true"))
    TSoftObjectPtr<UAnimSequence> SongAnimation = TSoftObjectPtr<UAnimSequence>(FSoftObjectPath(TEXT("/Script/Engine.AnimSequence'/Game/Assets/Animations/likejennie/jennie1_Anim.jennie1_Anim'")));
};
