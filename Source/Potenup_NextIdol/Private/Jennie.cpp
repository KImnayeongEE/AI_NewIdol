#include "Jennie.h"

#include "Animation/AnimSequence.h"
#include "Animation/AnimationAsset.h"
#include "Components/SkeletalMeshComponent.h"

AJennie::AJennie()
{
    PrimaryActorTick.bCanEverTick = false;
}

void AJennie::BeginPlay()
{
    Super::BeginPlay();

    if (bAutoStartOnBeginPlay)
    {
        SetActorLocation(SongStartLocation);
        StartSongPerformance();
    }
}

void AJennie::StartSongPerformance()
{
    if (USkeletalMeshComponent* MeshComponent = GetMesh())
    {
        MeshComponent->SetAnimationMode(EAnimationMode::AnimationSingleNode);

        if (UAnimSequence* LoadedAnim = SongAnimation.LoadSynchronous())
        {
            MeshComponent->PlayAnimation(LoadedAnim, true);
        }
    }
}

void AJennie::NotifySongStarted(int32 SongIndex)
{
    if (SongIndex == TargetSongIndex)
    {
        SetActorLocation(SongStartLocation);
        StartSongPerformance();
    }
}
